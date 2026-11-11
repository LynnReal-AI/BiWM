"""Restore and compare complete native H3 DMD state; perform zero optimizer updates."""
import argparse
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor

from h3.dmd.memory import enable_checkpoint_input_offload
from h3.dmd.score import FullScoreH3
from h3.dmd.state import capture_rng, config_digest, load_checkpoint
from h3.fa3_autograd import verify_fa3_backward
from h3.framepack import FramePackH3
from h3.h3_fa3_policy import require_h3_fa3_training_backend
from pipelines.h3.train_stage1 import checkpoint_context, configure_lora, wrap_fsdp
from pipelines.h3.train_stage2 import wrap
from pipelines.h3.train_stage3 import optimizer


def equal(left, right):
    if isinstance(left, ShardedTensor):
        assert isinstance(right, ShardedTensor) and left.size() == right.size() and left.dtype == right.dtype
        assert len(left.local_shards()) == len(right.local_shards())
        for a, b in zip(left.local_shards(), right.local_shards()):
            # Device relocation is the purpose of restore; shard identity,
            # shape, dtype and every value must remain identical.
            assert a.metadata.shard_offsets == b.metadata.shard_offsets
            assert a.metadata.shard_sizes == b.metadata.shard_sizes
            assert a.metadata.placement.rank() == b.metadata.placement.rank()
            equal(a.tensor, b.tensor)
    elif isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    checkpoint, output = Path(args.checkpoint).resolve(), Path(args.output).resolve()
    config = json.loads((checkpoint.parent / 'executed_config.json').read_text())
    local = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    require_h3_fa3_training_backend('_flash_3', device)
    fa3 = verify_fa3_backward(device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8 or 'H200' not in torch.cuda.get_device_name(device):
        raise ValueError('This full-native restore gate requires the saved8H200 topology')
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output/'config.json').write_text(json.dumps(dict(stage='complete DMD state restore gate',
            checkpoint=str(checkpoint), saved_config_sha256=config_digest(config), world_size=world,
            new_generator_updates=0, new_critic_updates=0, model_data_optimizer_contract=config), indent=2)+'\n')
    dist.barrier()
    phase2 = json.loads((Path(config['source']['phase2_checkpoint']).parent/'executed_config.json').read_text())
    generator = FramePackH3.strict_from_pretrained(phase2['initialization_path'], subfolder='transformer',
        torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
    generator.install_framepack()
    configure_lora(generator, argparse.Namespace(lora_rank=phase2['train']['lora_rank'],
                                                lora_alpha=phase2['train']['lora_alpha']))
    generator.history_compressor.requires_grad_(True)
    generator.set_attention_backend('_flash_3')
    enable_checkpoint_input_offload(generator)
    generator = wrap(generator, device)
    critic = FullScoreH3.strict_from_pretrained(config['source']['score_bundle'], subfolder='transformer',
        torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True).train().requires_grad_(True)
    critic.set_attention_backend('_flash_3')
    enable_checkpoint_input_offload(critic)
    critic = wrap_fsdp(critic, device)
    go, co = optimizer(generator, config['optimizer']['generator']), optimizer(critic, config['optimizer']['critic'])
    streams = {key: torch.Generator(device=device).manual_seed(config['rng'][key] + rank)
               for key in ('generator', 'critic', 'score', 'control')}
    counters = load_checkpoint(checkpoint, generator, critic, go, co, config, device, rng_streams=streams)
    original = torch.load(checkpoint/f'rank{rank:03d}.pt', map_location='cpu', weights_only=False)
    restored_rng = capture_rng(device, streams)
    # All ranks enter both FSDP state-dict collectives before local comparisons.
    with checkpoint_context(generator):
        restored_g = generator.state_dict()
    with checkpoint_context(critic):
        restored_c = critic.state_dict()
    error = None
    try:
        equal(original['generator'], restored_g)
        equal(original['critic'], restored_c)
        equal(original['generator_optimizer'], go.state_dict())
        equal(original['critic_optimizer'], co.state_dict())
        equal(original['rng'], restored_rng)
        equal(original['rng'], capture_rng(device, streams))
        equal(original['counters'], asdict(counters))
    except Exception as exc:
        error = type(exc).__name__
    errors = [None] * world
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError('Full-state restore comparison failed on ranks: ' + str(errors))
    result = dict(rank=rank, world_size=world, checkpoint=str(checkpoint),
        restored_counters=asdict(counters), model_G_C_shards_bitwise=True,
        two_FP32_Adam_states_bitwise=True, CPU_CUDA_Python_NumPy_named_RNG_bitwise=True,
        data_cursors_bitwise=True, new_generator_updates=0, new_critic_updates=0, FA3_gate=fa3,
        scope='Actual native H3 state restore only; no next optimizer step, generation or quality claim')
    (output/f'RESTORE.rank{rank:03d}.json').write_text(json.dumps(result, indent=2)+'\n')
    dist.barrier()
    if rank == 0:
        (output/'COMPLETE.json').write_text(json.dumps(dict(world_size=world, all_ranks_passed=True,
            checkpoint=str(checkpoint), new_generator_updates=0, new_critic_updates=0), indent=2)+'\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
