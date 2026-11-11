"""Full temporal-span DMD protocol on tiny CPU models, never a GPU run.

Exercises the planned native9chunk configuration through the medium-history
FramePack branch, including nonzero LoRA/compressor updates and next-cycle resume.
"""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random

import numpy as np
from peft import LoraConfig
import torch
import torch.distributed as dist

from h3.optimizer import FP32MasterAdamW
from pipelines.h3.train_stage1 import checkpoint_context, wrap_fsdp
from h3.framepack import FramePackH3
from pipelines.h3.train_stage2 import wrap
from h3_dmd_fixtures import equal, sample, tiny
from h3_dmd_restore import equal as equal_sharded
from h3.dmd.core import make_schedule, student_rollout
from h3.dmd.engine import Engine
from h3.dmd.score import FullScoreH3
from h3.dmd.state import Counters, capture_rng, load_checkpoint, save_checkpoint
from pipelines.h3.train_stage3 import check_config, restore


def main():
    dist.init_process_group('gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4
    device = torch.device('cpu')
    root = Path(os.environ['BIWM_DMD_CPU_OUT']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    draft = json.loads(Path(os.environ['BIWM_DMD_DRAFT']).read_text())
    check_config(draft)
    if rank == 0:
        (root / 'executed_config.json').write_text(json.dumps(draft, indent=2) + '\n')
    dist.barrier()
    assert draft['train']['max_generator_steps'] == 2 and draft['train']['max_critic_steps'] == 15
    torch.manual_seed(817)
    plain = tiny(FramePackH3, offload=True)
    suffixes = ('attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.to_out.0', 'ff.net.0.proj', 'ff.net.2')
    targets = [n for n, p in plain.named_modules() if isinstance(p, torch.nn.Linear) and n.endswith(suffixes)]
    plain.requires_grad_(False)
    # Tiny CPU rank4 is a fixture, not the production rank384 architecture.
    plain.add_adapter(LoraConfig(r=4, lora_alpha=4, target_modules=targets, lora_dropout=0., bias='none'))
    for name, parameter in plain.named_parameters():
        if '.lora_' in name:
            parameter.data = (.01 * torch.randn_like(parameter.float())).bfloat16()
    plain.history_compressor.requires_grad_(True)
    g = wrap(plain, device)
    c = wrap_fsdp(tiny(FullScoreH3, offload=True), device)
    teacher = wrap_fsdp(tiny(FullScoreH3).eval().requires_grad_(False), device)
    teacher_before = {k: v.detach().clone() for k, v in teacher.named_parameters()}
    def opt(model, role):
        cfg = draft['optimizer'][role]
        return FP32MasterAdamW((p for p in model.parameters() if p.requires_grad),
            lr=cfg['lr'], betas=tuple(cfg['betas']), eps=cfg['eps'], weight_decay=cfg['weight_decay'])
    go, co = opt(g, 'generator'), opt(c, 'critic')
    values = sample()
    values['target_latents'] = torch.randn(24, 47, 2, 2).bfloat16()
    values['camera_viewmats'] = torch.eye(4).repeat(47, 1, 1)
    values['camera_viewmats'][:, 0, 3] = torch.linspace(0, 2, 47)
    values['camera_K'] = torch.eye(3).repeat(47, 1, 1)
    schedule = make_schedule(draft['dmd']['start_sigmas'], draft['dmd']['shift'])
    with torch.no_grad():
        rng = torch.Generator().manual_seed(1234 + rank)
        generated, _ = student_rollout(g, values, schedule, 9, 0, rng=rng, train_mode=False)
        changed = dict(values, target_latents=torch.full_like(values['target_latents'], 999.))
        again, _ = student_rollout(g, changed, schedule, 9, 0,
            rng=torch.Generator().manual_seed(1234 + rank), train_mode=False)
    assert generated.shape == (1, 24, 47, 2, 2)
    equal(generated, again)
    streams = {key: torch.Generator().manual_seed(draft['rng'][key] + rank)
               for key in ('generator', 'critic', 'score', 'control')}
    seen = []
    def reader(role, absolute):
        identity = f'{role}:{absolute}'
        seen.append(identity)
        return identity, values
    engine = Engine(g, teacher, c, go, co, Counters(critic_warmup=5, effective_batch=8),
        reader, schedule, draft['dmd'], device, streams)
    random.seed(221 + rank)
    np.random.seed(231 + rank)
    torch.manual_seed(241 + rank)
    for _ in range(5):
        receipt = engine.update('critic')
        assert receipt['generator'] == 0 and receipt['latent_frames'] == 47
    engine.counters.validate(boundary=True)
    assert engine.counters.critic == 5
    assert not go.state  # Warmup must not update the generator or its Adam.
    first_C, first_G = engine.cycle()
    assert first_G['generator'] == 1 and first_G['critic'] == 10
    assert first_G['latent_frames'] == 47 and first_G['blocks'] == 9
    assert first_G['grad_norm'] > 0
    # Verify actual learned far-history s8 compressor participation, not merely
    # a47-frame output shape. All local FSDP shards contribute to this sum.
    far = torch.zeros((), dtype=torch.float64)
    for name, parameter in g.named_parameters():
        if 'history_compressor' in name and '.spatial.8.' in name and parameter in go.state:
            far += go.state[parameter]['exp_avg'].double().abs().sum()
    dist.all_reduce(far)
    assert float(far) > 0, 'Native47-latent rollout did not train the medium-history s8 compressor'
    checkpoint = save_checkpoint(root, g, c, go, co, engine.counters, draft,
                                 device, rng_streams=streams)
    def snapshot():
        return dict(g={k: v.detach().clone() for k, v in g.named_parameters()},
            c={k: v.detach().clone() for k, v in c.named_parameters()},
            go=copy.deepcopy(go.state_dict()), co=copy.deepcopy(co.state_dict()),
            streams={k: v.get_state().clone() for k, v in streams.items()},
            torch_rng=torch.get_rng_state().clone(), counters=asdict(engine.counters))
    final_C, final_G = engine.cycle()
    assert final_G['generator'] == 2 and final_G['critic'] == 15
    assert final_G['generator_samples'] == 16 and final_G['critic_samples'] == 120
    expected = snapshot()
    torch.save(expected, root / f'FINAL_STATE.rank{rank:03d}.pt')
    gathered = [None] * world
    dist.all_gather_object(gathered, seen)
    used = [identity for identities in gathered for identity in identities]
    assert sorted(z for z in used if z.startswith('critic:')) == sorted(f'critic:{i}' for i in range(120))
    assert sorted(z for z in used if z.startswith('generator:')) == sorted(f'generator:{i}' for i in range(16))
    engine.counters = load_checkpoint(checkpoint, g, c, go, co, draft, device, rng_streams=streams)
    replay_C, replay_G = engine.cycle()
    equal(expected, snapshot())
    for left, right in zip(final_C + [final_G], replay_C + [replay_G]):
        for key in ('sample_ids', 'sample_cursor_before', 'grad_exits', 'score_sigmas',
                    'loss', 'global_loss', 'DMD_direction_abs_mean', 'grad_norm'):
            equal(left[key], right[key])
    # Exercise the real entrypoint's cross-run restore guard, with the old
    # checkpoint digest and changed descriptive/review/budget metadata only.
    next_root = root / 'continuation_probe'
    next_config = copy.deepcopy(draft)
    next_config.update(continuation=str(checkpoint), status='CPU reviewed continuation probe',
                       checkpoint_steps=[2], validation_steps=[2])
    next_config['train']['initial_generator_step'] = 1
    next_config['source']['teacher_quality_review'] = 'CPU-only-new-review-metadata'
    check_config(next_config)
    engine.counters = restore(next_root, next_config, g, c, go, co, device, streams)
    saved = torch.load(checkpoint / f'rank{rank:03d}.pt', map_location='cpu', weights_only=False)
    with checkpoint_context(g):
        restored_g = g.state_dict()
    with checkpoint_context(c):
        restored_c = c.state_dict()
    equal_sharded(saved['generator'], restored_g)
    equal_sharded(saved['critic'], restored_c)
    equal_sharded(saved['generator_optimizer'], go.state_dict())
    equal_sharded(saved['critic_optimizer'], co.state_dict())
    equal_sharded(saved['rng'], capture_rng(device, streams))
    equal_sharded(saved['counters'], asdict(engine.counters))
    del saved, restored_g, restored_c
    continued_C, continued_G = engine.cycle()
    equal(expected, snapshot())
    for left, right in zip(final_C + [final_G], continued_C + [continued_G]):
        for key in ('sample_ids', 'sample_cursor_before', 'grad_exits', 'score_sigmas',
                    'loss', 'global_loss', 'DMD_direction_abs_mean', 'grad_norm'):
            equal(left[key], right[key])
    for name, parameter in teacher.named_parameters():
        equal(teacher_before[name], parameter.detach())
        assert parameter.grad is None
    result = dict(rank=rank, world_size=world, draft_configuration_checked=True,
        native_temporal_span_47latent_9chunks=True, generated_history_no_GT_leak=True,
        tiny_nonzero_LoRA_and_compressor_generator=True,
        far_history_scale8_optimizer_moment_abs_sum=float(far),
        actual_warmup5_then_2G_15C=True, full_native_tiny_critic=True,
        actual_global_C120_G16_sample_positions=True,
        teacher_CFG=draft['dmd']['real_guidance_scale'],
        shift3_clip5_betas0_999=True,
        critic_optimizer_offload_for_generator=draft['dmd'].get('critic_optimizer_offload_for_generator', False),
        resumed_next_5C1G_parameters_FP32_Adam_RNG_receipts_bitwise=True,
        entrypoint_crossrun_reviewed_metadata_restore_nextcycle_bitwise=True,
        restore_only_comparator_CPU_sharded_models_Adams_RNG_counters_bitwise=True,
        teacher_frozen_no_gradient=True,
        scope='4rank CPU tinyH3/rank4-LoRA/spatial2x2 with productionEngine; not33B/rank384/H200, memory, decodedRGB, convergence or launchedDMD')
    (root / f'CPU_GATE.rank{rank:03d}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
