# SPDX-License-Identifier: Apache-2.0
"""Native full-video teacher/phase2-AR comparisons before any DMD update.

Requires the completed and selected phase2 run plus its score export. This is
an explicit inference command, never a side effect of a progress inspection.
"""
import argparse
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from h3.data import CameraDataset
from h3.fa3_autograd import verify_fa3_backward
from h3.h3_fa3_policy import require_h3_fa3_training_backend
from pipelines.h3.train_stage1 import wrap_fsdp
from pipelines.h3.train_stage2 import deliver
from h3.dmd.score import FullScoreH3, sample_score
from pipelines.h3.train_stage3 import rank0_check, verify_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Teacher evaluation requires a new output directory')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    require_h3_fa3_training_backend('_flash_3', device)
    gate = verify_fa3_backward(device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world not in config['allowed_world_sizes'] or world not in (4, 8):
        raise ValueError('Only the authorized4/8H200 teacher evaluation is supported')
    if 'H200' not in torch.cuda.get_device_name(device):
        raise ValueError('Native teacher evaluation requires H200')
    p2, complete = rank0_check(lambda: verify_sources(config))
    indices = config['validation']['indices']
    if len(indices) != 8 or len(set(indices)) != 8:
        raise ValueError('Require the same eight held-out phase2 cases')
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        executed = dict(config)
        executed.update(actual_world_size=world, DP=world, SP=1, attention='_flash_3',
                        stage='BiWM H3 phase3 native teacher preflight', G_updates=0, C_updates=0)
        (output / 'executed_config.json').write_text(json.dumps(executed, indent=2) + '\n')
    dist.barrier()
    (output / f'FA3_GATE.rank{rank:03d}.json').write_text(json.dumps(gate, indent=2) + '\n')
    model = FullScoreH3.strict_from_pretrained(config['source']['score_bundle'], subfolder='transformer',
        torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True).eval().requires_grad_(False)
    model.set_attention_backend('_flash_3')
    model = wrap_fsdp(model, device)
    validation = CameraDataset(config['data']['validation_manifest'])
    cases = indices[rank::world]
    fixed = {case: {k: v.to(device) for k, v in validation[case].items()} for case in cases}
    source = Path(config['source']['phase2_checkpoint']).resolve().parent
    step = complete['step']
    selected_validation = source if step == p2['train']['max_steps'] else source / f'validation_step{step:06d}'
    sampling = config['sampling']
    metrics = []
    for case in cases:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        result = sample_score(model, fixed[case], points=sampling['points'], shift=sampling['shift'],
                              guidance_scale=sampling['guidance_scale'], seed=42 + case)
        torch.cuda.synchronize(device)
        if result.shape != (1, 24, 47, 48, 84):
            raise ValueError('Teacher output must cover native158f768p')
        torch.save(result.cpu(), output / f'step{step}.case{case:03d}.pt')
        metrics.append(dict(case_id=case, rank=rank, points=sampling['points'],
            teacher_CFG=sampling['guidance_scale'], seconds=time.monotonic() - started,
            peak_GiB=torch.cuda.max_memory_allocated(device) / 2**30,
            source_weight='Live', source_step=step, G_updates=0, C_updates=0))
        del result
    del model
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    for case in cases:
        before = torch.load(selected_validation / f'step{step}.case{case:03d}.pt', map_location='cpu', weights_only=True)
        current = torch.load(output / f'step{step}.case{case:03d}.pt', map_location='cpu', weights_only=True)
        deliver(p2['model_base'], output, fixed[case], before, current, case, device,
                rank=rank, step=step, baseline_step=step, model_label='Native full-video teacher',
                baseline_label=f'Phase2 FramePack AR Live{step}')
        marker = output / 'validation' / f'case{case:03d}' / 'COMPLETE.json'
        record = json.loads(marker.read_text())
        record.update(role='native_full_video_teacher', source_phase2_step=step,
            sampler_points=sampling['points'], guidance_scale=sampling['guidance_scale'],
            G_updates=0, C_updates=0, scope='Full bidirectional teacher versus phase2 AR; not a DMD update or quality acceptance')
        marker.write_text(json.dumps(record, indent=2) + '\n')
    (output / f'TEACHER_GATE.rank{rank:03d}.json').write_text(json.dumps(dict(metrics=metrics,
        scope='Actual native teacher inference/media integrity, not a DMD update or quality acceptance'), indent=2) + '\n')
    dist.barrier()
    if rank == 0:
        (output / 'COMPLETE.json').write_text(json.dumps(dict(cases=8,frames=158,source_step=step,
            G_updates=0,C_updates=0,quality_review_pending=True,source_weight='Live'), indent=2) + '\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
