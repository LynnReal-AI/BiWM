# SPDX-License-Identifier: Apache-2.0
"""Explicit phase2 Live transfer and CPU FP32 merge for full native scores."""
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType

from pipelines.h3.train_stage1 import checkpoint_context
from h3.framepack_checkpoint import verify_checkpoint_files, verify_finished_validation
from .score import FullScoreH3
from .state import file_digest


def require_phase2_selection(checkpoint, selection_path, world):
    """An explicit reviewed158f decision, not STATUS or an inherited pilot."""
    checkpoint = Path(checkpoint).resolve()
    complete = verify_checkpoint_files(checkpoint, world)
    source = checkpoint.parent
    config_path = source / 'executed_config.json'
    config = json.loads(config_path.read_text())
    selection = json.loads(Path(selection_path).read_text())
    if (Path(selection['checkpoint']).resolve() != checkpoint
        or selection['source_config_sha256'] != file_digest(config_path)
        or selection['source_weight'] != 'Live' or not selection['accepted_for_DMD']):
        raise ValueError('Phase2 selection/source/quality decision differs')
    if config['validation']['rollout_latents'] != 47 or config['validation']['rollout_rgb_frames'] != 158:
        raise ValueError('DMD requires the native158f phase2 validation, not39f pilot')
    if config['validation']['cases'] != 8 or len(selection['reviewed_cases']) != 8:
        raise ValueError('All eight phase2 validation cases must be reviewed')
    cases = config['validation'].get('indices', list(range(8)))
    if sorted(selection['reviewed_cases']) != sorted(cases) or selection['model_step'] != complete['step']:
        raise ValueError('Quality decision belongs to different cases/model step')
    # Finish the bounded phase2 run and deliver its final cases first. Source
    # selection may then choose a better intermediate checkpoint with its own
    # matched native snapshot instead of silently forcing the latest weights.
    final_step = config['train']['max_steps']
    verify_checkpoint_files(source / f'checkpoint_{final_step:06d}', world)
    verify_finished_validation(source, config, final_step, world)
    if complete['step'] != final_step:
        if complete['step'] > final_step:
            raise ValueError('Selected checkpoint exceeds the completed phase2 budget')
        verify_finished_validation(source, config, complete['step'], world,
            validation_root=source / f"validation_step{complete['step']:06d}")
    return config, complete


def load_phase2_generator(checkpoint, model, world, device):
    """Preserve phase2 LoRA+compressor exactly; DMD optimizers start separately."""
    checkpoint = Path(checkpoint)
    complete = verify_checkpoint_files(checkpoint, world)
    rank = dist.get_rank()
    state = torch.load(checkpoint / f'rank{rank:03d}.pt', map_location='cpu', weights_only=False)
    if state['world_size'] != world or state['step'] != complete['step']:
        raise ValueError('Phase2 rank payload differs from complete marker')
    with checkpoint_context(model):
        model.load_state_dict(state['model'], strict=True)
    # This is a new objective/phase, not resume of the FM optimizer/RNG/data.
    del state
    dist.barrier()
    return complete['step']


def merged_score_state(full, *, rank, alpha, expected_targets):
    """Separate compressor and fold every phase2 LoRA in FP32 on CPU.

    Preserve the original mixed dtype at each merged weight. Input tensors are
    never modified; this is export, not mutation of a live generator model.
    """
    suffix = '.lora_A.default.weight'
    names = sorted(key[:-len(suffix)] for key in full if key.endswith(suffix))
    if len(names) != expected_targets or rank < 1 or alpha <= 0:
        raise ValueError('Phase2 LoRA target count/rank/alpha differs')
    merged, consumed = {}, set()
    for name in names:
        a_key, b_key = name + suffix, name + '.lora_B.default.weight'
        w_key, bias_key = name + '.base_layer.weight', name + '.base_layer.bias'
        a, b, weight = full[a_key], full[b_key], full[w_key]
        if any(value.device.type != 'cpu' for value in (a, b, weight)):
            raise ValueError('Phase2 LoRA must merge on CPU, never gather33B on one GPU')
        if a.shape != (rank, weight.shape[1]) or b.shape != (weight.shape[0], rank):
            raise ValueError('Phase2 LoRA pair/base geometry differs')
        value = weight.float() + (b.float() @ a.float()) * (alpha / rank)
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite phase2 LoRA merge')
        merged[name + '.weight'] = value.to(weight.dtype).contiguous()
        consumed.update((a_key, b_key, w_key))
        if bias_key in full:
            merged[name + '.bias'] = full[bias_key]
            consumed.add(bias_key)
    compressor = {}
    for key, value in full.items():
        if key in consumed:
            continue
        if key.startswith('history_compressor.'):
            compressor[key.removeprefix('history_compressor.')] = value
        elif '.lora_' in key or '.base_layer.' in key:
            raise ValueError('Unrecognized adapter/base key in phase2 export: ' + key)
        else:
            merged[key] = value
    if not compressor:
        raise ValueError('FramePack phase2 compressor is missing')
    return merged, compressor


def export_score_bundle(generator, config, output, *, source_checkpoint):
    """Full-state CPU gather, native score weights, and separate compressor.

    Call only after selecting and loading the phase2 checkpoint. The generator
    retains its exact unfused LoRA state; teacher/fake load the merged native
    weights without a history compressor. No optimizer or GPU job is started.
    """
    output = Path(output)
    if output.exists():
        raise ValueError('Score export must use a new directory')
    # CPU fixtures are already on CPU. Redundant FSDP CPU-offload of a CPU
    # full-state gather segfaults in this installed torch2.9.1 build. Production
    # CUDA parameters still explicitly offload the full gather to CPU.
    needs_offload = next(generator.parameters()).device.type == 'cuda'
    with FSDP.state_dict_type(generator, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=needs_offload, rank0_only=True)):
        full = generator.state_dict()
    error = [None]
    if dist.get_rank() == 0:
        try:
            from accelerate import init_empty_weights
            from safetensors.torch import save_file
            weights, compressor = merged_score_state(full, rank=config['train']['lora_rank'],
                alpha=config['train']['lora_alpha'], expected_targets=config['train']['lora_targets'])
            with init_empty_weights():
                score = FullScoreH3.from_config(generator.module.config)
            score.load_state_dict(weights, strict=True, assign=True)
            output.mkdir(parents=True)
            score.save_pretrained(output / 'transformer', safe_serialization=True)
            save_file({k: v.contiguous() for k, v in compressor.items()}, str(output / 'framepack.safetensors'))
            adapters = {k: v.contiguous() for k, v in full.items() if '.lora_' in k}
            if len(adapters) != 2 * config['train']['lora_targets']:
                raise ValueError('Portable generator adapter count differs')
            save_file(adapters, str(output / 'generator_adapter.safetensors'))
            files = [dict(path=str(p.relative_to(output)), bytes=p.stat().st_size, sha256=file_digest(p))
                     for p in sorted(output.rglob('*')) if p.is_file()]
            receipt = dict(source_checkpoint=str(Path(source_checkpoint).resolve()), source_weight='Live',
                stage='phase2 Live -> phase3 initialization; not an optimizer resume',
                merged_modules=config['train']['lora_targets'], precision='FP32 merge then native weight dtype',
                generator='Original phase2 LoRA tensors plus compressor; load on the identical official camera-merged base',
                generator_base=config.get('initialization_path'),
                teacher='Frozen native full-video score, no FramePack',
                critic='Independent trainable native full-video score; no reduced LoRA-only critic', files=files)
            (output / 'COMPLETE.json').write_text(json.dumps(receipt, indent=2) + '\n')
        except Exception as exc:
            error[0] = f'{type(exc).__name__}: {exc}'
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError('Cannot export selected phase2 score bundle: ' + error[0])
    dist.barrier()


def load_portable_generator(model, bundle):
    """Load exact unfused phase2 adapters/compressor on its unchanged frozen base.

    The caller verifies the bundle SHA/source and the original camera-merge
    base before constructing this model. This is world-size-independent and
    does not inherit an FM optimizer or quantize/fuse the generator adapter.
    """
    from safetensors.torch import load_file
    bundle = Path(bundle)
    adapter = load_file(str(bundle / 'generator_adapter.safetensors'))
    expected = {k for k in model.state_dict() if '.lora_' in k}
    if set(adapter) != expected:
        raise ValueError('Portable generator LoRA key set differs')
    current = model.state_dict()
    if any(adapter[k].dtype != current[k].dtype or adapter[k].shape != current[k].shape for k in expected):
        raise ValueError('Portable generator LoRA dtype/shape differs')
    result = model.load_state_dict(adapter, strict=False)
    if result.unexpected_keys or set(result.missing_keys) != set(model.state_dict()) - expected:
        raise ValueError('Portable generator LoRA load differs')
    model.history_compressor.load_state_dict(load_file(str(bundle / 'framepack.safetensors')), strict=True)
