# SPDX-License-Identifier: Apache-2.0
"""Explicit same-world phase2 continuation; never relax in-place resume checks."""
import hashlib
import json
from pathlib import Path
import subprocess

import torch.distributed as dist

from pipelines.h3.train_stage1 import resume_checkpoint


def phase2_contract(config):
    """Fields that must remain identical when extending this phase2 training."""
    return dict(
        **{key: config[key] for key in (
            'model_base', 'adapter', 'initialization_path', 'conditions', 'exclusions',
            'objective', 'optimizer')},
        train={key: config['train'][key] for key in (
            'gpu', 'FSDP', 'SP', 'micro_batch', 'effective_batch', 'lora_rank',
            'lora_alpha', 'lora_targets', 'train_compressor', 'attention_backend', 'seed')},
        data={key: config['data'][key] for key in (
            'qwen_skip_reference_videos', 'resolution', 'pixel_frames', 'latent_frames')})


def verify_checkpoint_files(checkpoint, world):
    checkpoint = Path(checkpoint)
    complete = json.loads((checkpoint / 'COMPLETE.json').read_text())
    if complete['world_size'] != world:
        raise ValueError('Continuation requires the saved FSDP world size')
    for rank in range(world):
        if not (checkpoint / f'rank{rank:03d}.pt').is_file() or (checkpoint / f'rank{rank:03d}.pt').stat().st_size == 0:
            raise ValueError('Incomplete checkpoint rank files')
    return complete


def verify_finished_validation(source, config, step, world):
    source = Path(source)
    for rank in range(world):
        receipt = json.loads((source / 'step_receipts' / f'COMPLETE.step{step:06d}.rank{rank:03d}.json').read_text())
        if receipt['step'] != step or receipt['rank'] != rank or receipt['world_size'] != world:
            raise ValueError('Source step is not complete on all ranks')
    cases = config['validation'].get('indices', list(range(config['validation']['cases'])))
    for case in cases:
        directory = source / 'validation' / f'case{case:03d}'
        complete = json.loads((directory / 'COMPLETE.json').read_text())
        if not complete.get('full_decode') or complete['case_id'] != case:
            raise ValueError('Source validation case is incomplete')
        if complete.get('model_step', step) != step:
            raise ValueError('Source validation belongs to another model step')
        names = ('baseline.mp4', f'step{step}.mp4', f'GT_step{step}.mp4', f'GT_baseline_step{step}.mp4')
        for name, panels in zip(names, (1, 1, 2, 3)):
            if not (directory / name).is_file() or (directory / name).stat().st_size == 0:
                raise ValueError('Source validation MP4 is missing')
            probe = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_read_frames,width,height', '-of', 'json', str(directory / name)]))['streams'][0]
            height, width = config['data']['resolution']
            if (int(probe['nb_read_frames']), probe['height'], probe['width']) != (
                    config['validation']['rollout_rgb_frames'], height, width * panels):
                raise ValueError('Source validation MP4 geometry/frame count differs')
            subprocess.run(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(directory / name),
                            '-f', 'null', '-'], check=True, stdout=subprocess.DEVNULL)


def restore_training(output, model, optimizer, config, rank, world, device):
    output = Path(output)
    local = sorted(path for path in output.glob('checkpoint_*') if (path / 'COMPLETE.json').exists())
    if local:
        verify_checkpoint_files(local[-1], world)
        return resume_checkpoint(local[-1], model, optimizer, config, rank, world, device), True
    continuation = config.get('continuation')
    if not continuation:
        return 0, False
    checkpoint = Path(continuation['checkpoint']).resolve()
    complete = verify_checkpoint_files(checkpoint, world)
    source_config_path = checkpoint.parent / 'executed_config.json'
    source_config = json.loads(source_config_path.read_text())
    if phase2_contract(source_config) != phase2_contract(config):
        raise ValueError('Continuation would change the phase2 model/loss/optimizer/native input contract')
    if (source_config['data']['train_manifest_sha256'] == config['data']['train_manifest_sha256']
        and source_config['train'].get('sample_step_origin', 0) != config['train'].get('sample_step_origin', 0)):
        raise ValueError('The unchanged training manifest must retain its sample cursor origin')
    # Switching the pilot subset to full data requires actual completed pilot
    # optimizer steps AND delivered validation. A draft/STATUS flag is not enough.
    needs_validation = bool(continuation.get('require_validation', False))
    if not source_config['data'].get('train_manifest') and config['data'].get('train_manifest'):
        if complete['step'] != source_config['train']['max_steps']:
            raise ValueError('Full-data continuation requires the finished short trial')
        needs_validation = True
    if needs_validation:
        error = [None]
        if rank == 0:
            try:
                verify_finished_validation(checkpoint.parent, source_config, complete['step'], world)
            except Exception as exc:
                error[0] = f'{type(exc).__name__}: {exc}'
        if dist.is_initialized():
            dist.broadcast_object_list(error, src=0, device=device)
        if error[0]:
            raise ValueError('Source validation failed: ' + error[0])
    # Passing the exact source config preserves the public strict checkpoint
    # check. Only this explicit transfer path permits recorded data/budget changes.
    step = resume_checkpoint(checkpoint, model, optimizer, source_config, rank, world, device)
    if rank == 0:
        receipt = dict(checkpoint=str(checkpoint), source_step=step, world_size=world,
                       source_config_sha256=hashlib.sha256(source_config_path.read_bytes()).hexdigest(),
                       optimizer_and_RNG_restored=True, phase2_contract_unchanged=True,
                       sample_step_origin=config['train'].get('sample_step_origin', 0))
        (output / 'CONTINUATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return step, False
