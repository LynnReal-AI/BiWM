# SPDX-License-Identifier: Apache-2.0
"""Config-driven camera-only H3 DMD; entry requires a reviewed phase2 export."""
import argparse
from dataclasses import asdict
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from h3.data import CameraDataset
from h3.fa3_autograd import verify_fa3_backward
from h3.h3_fa3_policy import require_h3_fa3_training_backend
from h3.optimizer import FP32MasterAdamW
from pipelines.h3.train_stage1 import configure_lora, wrap_fsdp
from h3.framepack_data import GlobalSampleOrder, SolarWMCameraDataset
from h3.framepack import FramePackH3
from pipelines.h3.train_stage2 import deliver, wrap
from pipelines.h3.configure_stage3 import continuation_contract
from h3.dmd.core import make_schedule, student_rollout
from h3.dmd.engine import Engine
from h3.dmd.initialization import load_portable_generator, require_phase2_selection
from h3.dmd.memory import enable_checkpoint_input_offload
from h3.dmd.score import FullScoreH3
from h3.dmd.state import Counters, file_digest, load_checkpoint, save_checkpoint


def check_config(config):
    t, data, dmd, val = (config[k] for k in ('train', 'data', 'dmd', 'validation'))
    if any(type(t[k]) is not int or t[k] < 0 for k in (
            'initial_generator_step', 'max_generator_steps', 'max_critic_steps', 'critic_warmup')):
        raise ValueError('Update budgets/counts must be nonnegative integers')
    if config['objectives'] != ['DMD', 'critic_FM'] or t['critic_capacity'] != 'full':
        raise ValueError('This route requires pure DMD plus a full native video critic')
    if t['attention_backend'] != '_flash_3' or t['SP'] != 1 or t['effective_batch'] != 8:
        raise ValueError('Require FA3, SP1 and effective batch8')
    if t['micro_batch'] != 1 or not set(t['allowed_world_sizes']) <= {4, 8}:
        raise ValueError('Only one-video microbatches on4/8H200 are authorized')
    if t['max_generator_steps'] <= t['initial_generator_step'] or t['critic_warmup'] < 0:
        raise ValueError('Require an explicit positive new-update budget')
    if t['max_critic_steps'] != t['critic_warmup'] + 5 * t['max_generator_steps']:
        raise ValueError('Critic budget must include warmup and exactly5C perG')
    if data['latent_frames'] != 47 or data['pixel_frames'] != 158 or data['resolution'] != [768, 1344]:
        raise ValueError('BiWM H3 production DMD uses full native158f768p camera data')
    if not data['qwen_skip_reference_videos'] or dmd['blocks'] != 9:
        raise ValueError('Require first-image-only Qwen and all9 native chunks')
    if len(dmd['start_sigmas']) != 4 or not math.isfinite(dmd['real_guidance_scale']):
        raise ValueError('Require explicit four-map rollout and finite teacher CFG')
    if type(dmd.get('critic_optimizer_offload_for_generator', False)) is not bool:
        raise ValueError('Critic optimizer CPU placement must be an explicit boolean')
    if type(dmd.get('generator_chunk_replay', False)) is not bool:
        raise ValueError('Generator chunk replay must be an explicit boolean')
    make_schedule(dmd['start_sigmas'], dmd['shift'])
    sigma = dmd['score_sigma']
    if not (0 <= sigma['min'] <= sigma['max'] <= 1 and sigma['shift'] > 0):
        raise ValueError('Invalid score sigma configuration')
    if val['latent_frames'] != 47 or val['rgb_frames'] != 158 or len(set(val['indices'])) != 8:
        raise ValueError('Require eight fixed native158f validation cases')
    first, final = t['initial_generator_step'] + 1, t['max_generator_steps']
    if not {first, final} <= set(config['checkpoint_steps']) or not {first, final} <= set(config['validation_steps']):
        raise ValueError('Save and validate the first newG and finalG')
    for role in ('generator', 'critic'):
        opt = config['optimizer'][role]
        if set(opt) != {'lr', 'betas', 'eps', 'weight_decay'} or opt['lr'] <= 0:
            raise ValueError('Explicit FP32-master AdamW hyperparameters required')
        if not math.isfinite(dmd['clip'][role]) or dmd['clip'][role] <= 0:
            raise ValueError('Explicit finite gradient clip required')


def rank0_check(function):
    result = [None, None]
    if dist.get_rank() == 0:
        try:
            result[0] = function()
        except Exception as exc:
            result[1] = f'{type(exc).__name__}: {exc}'
    dist.broadcast_object_list(result, src=0)
    if result[1]:
        raise ValueError(result[1])
    return result[0]


def verify_sources(config):
    source = config['source']
    checkpoint = Path(source['phase2_checkpoint']).resolve()
    source_world = json.loads((checkpoint / 'COMPLETE.json').read_text())['world_size']
    p2, complete = require_phase2_selection(checkpoint, source['selection'], source_world)
    if file_digest(source['selection']) != source['selection_sha256']:
        raise ValueError('Phase2 quality selection changed')
    bundle = Path(source['score_bundle']).resolve()
    if file_digest(bundle / 'COMPLETE.json') != source['score_bundle_sha256']:
        raise ValueError('Score initialization manifest changed')
    manifest = json.loads((bundle / 'COMPLETE.json').read_text())
    if Path(manifest['source_checkpoint']).resolve() != checkpoint or manifest['source_weight'] != 'Live':
        raise ValueError('Score export belongs to another phase2 source')
    if Path(manifest['generator_base']).resolve() != Path(p2['initialization_path']).resolve():
        raise ValueError('Unfused generator base differs from its phase2 source')
    if file_digest(Path(p2['initialization_path']) / 'MERGE_COMPLETE.json') != source['camera_merge_sha256']:
        raise ValueError('Official SolarWM camera initialization changed')
    for item in manifest['files']:
        path = (bundle / item['path']).resolve()
        if not path.is_relative_to(bundle) or path.stat().st_size != item['bytes'] or file_digest(path) != item['sha256']:
            raise ValueError('Score bundle file missing or changed: ' + item['path'])
    for key in ('train', 'validation'):
        if file_digest(config['data'][key + '_manifest']) != config['data'][key + '_manifest_sha256']:
            raise ValueError('DMD dataset manifest changed')
    if config['data']['train_manifest_sha256'] != p2['data']['train_manifest_sha256']:
        raise ValueError('DMD must retain the selected SolarWM training split')
    if (config['data']['validation_manifest_sha256'] != file_digest(Path(p2['data_path']) / 'validation.jsonl')
        or config['validation']['indices'] != p2['validation']['indices']):
        raise ValueError('DMD must retain the selected eight phase2 validation cases')
    return p2, complete


def optimizer(model, config):
    return FP32MasterAdamW((p for p in model.parameters() if p.requires_grad),
        lr=config['lr'], betas=tuple(config['betas']), eps=config['eps'], weight_decay=config['weight_decay'])


def restore(output, config, g, c, go, co, device, streams):
    own = sorted(p for p in output.glob('checkpoint_*') if (p / 'COMPLETE.json').exists())
    path = own[-1] if own else Path(config['continuation']) if config.get('continuation') else None
    if path is None:
        if any((output / 'step_receipts').glob('*.json')):
            raise ValueError('Prior updates have no complete checkpoint; use an explicit new recovery run')
        return Counters(critic_warmup=config['train']['critic_warmup'], effective_batch=8)
    saved_config = json.loads((path.parent / 'executed_config.json').read_text())
    if not own:
        if continuation_contract(saved_config) != continuation_contract(config):
            raise ValueError('DMD continuation changes model/data/optimizer/rollout contract')
    complete = json.loads((path / 'COMPLETE.json').read_text())['counters']
    if not own and complete['generator'] != config['train']['initial_generator_step']:
        raise ValueError('Continuation sourceG differs from the configured budget origin')
    for prefix, limit in (('COMPLETE', complete['generator']), ('CRITIC', complete['critic'])):
        for receipt in (output / 'step_receipts').glob(prefix + '.step*.rank*.json'):
            if int(receipt.name.split('.')[1].removeprefix('step')) > limit:
                raise ValueError('Unsaved later updates exist; recover explicitly into a new run')
    counters = load_checkpoint(path, g, c, go, co, saved_config if not own else config,
                               device, rng_streams=streams)
    return counters


def main():
    output = Path(os.environ['BIWM_RUN']).resolve()
    config = json.loads((output / 'config.json').read_text())
    check_config(config)
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    require_h3_fa3_training_backend('_flash_3', device)
    fa3 = verify_fa3_backward(device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world not in config['train']['allowed_world_sizes'] or 'H200' not in torch.cuda.get_device_name(device):
        raise ValueError('Only the authorized4/8H200 topology may run DMD')
    config['train'].update(world_size=world, DP=world, accumulation=8 // world)
    def write_executed():
        path = output / 'executed_config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('Refuse to overwrite an existing run execution contract')
        path.write_text(json.dumps(config, indent=2) + '\n')
    rank0_check(write_executed)
    (output / f'FA3_GATE.rank{rank:03d}.json').write_text(json.dumps(fa3, indent=2) + '\n')
    def status(phase, **extra):
        if rank == 0:
            (output / 'STATUS.json').write_text(json.dumps(dict(phase=phase, at=time.time(), **extra)) + '\n')
    status('verifying_selected_phase2')
    p2, source = rank0_check(lambda: verify_sources(config))
    bundle = Path(config['source']['score_bundle'])
    torch.manual_seed(config['seed'] + rank)
    g = FramePackH3.strict_from_pretrained(p2['initialization_path'], subfolder='transformer',
        torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
    g.install_framepack()
    configure_lora(g, argparse.Namespace(lora_rank=p2['train']['lora_rank'], lora_alpha=p2['train']['lora_alpha']))
    load_portable_generator(g, bundle)
    g.history_compressor.requires_grad_(True)
    g.set_attention_backend('_flash_3')
    enable_checkpoint_input_offload(g)
    g = wrap(g, device)
    status('loading_full_native_scores')
    teacher = FullScoreH3.strict_from_pretrained(bundle, subfolder='transformer', torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True).eval().requires_grad_(False)
    teacher.set_attention_backend('_flash_3')
    teacher = wrap_fsdp(teacher, device)
    c = FullScoreH3.strict_from_pretrained(bundle, subfolder='transformer', torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True).train().requires_grad_(True)
    c.set_attention_backend('_flash_3')
    enable_checkpoint_input_offload(c)
    c = wrap_fsdp(c, device)
    go, co = optimizer(g, config['optimizer']['generator']), optimizer(c, config['optimizer']['critic'])
    torch.manual_seed(config['seed'] + rank)
    random.seed(config['seed'] + rank)
    np.random.seed(config['seed'] + rank)
    streams = {key: torch.Generator(device=device).manual_seed(config['rng'][key] + rank)
               for key in ('generator', 'critic', 'score', 'control')}
    counters = restore(output, config, g, c, go, co, device, streams)
    if counters.generator > config['train']['max_generator_steps']:
        raise ValueError('RestoredG exceeds the explicit update budget')
    data = SolarWMCameraDataset(config['data']['train_manifest'], config['data']['release_root'])
    order = {role: GlobalSampleOrder(len(data), config['data'][role + '_seed']) for role in ('generator', 'critic')}
    def reader(role, absolute):
        index = order[role](absolute)
        return data.rows[index]['sample_id'], {k: v.to(device) for k, v in data[index].items()}
    val = CameraDataset(config['data']['validation_manifest'])
    cases = config['validation']['indices'][rank::world]
    fixed = {case: {k: v.to(device) for k, v in val[case].items()} for case in cases}
    schedule = make_schedule(config['dmd']['start_sigmas'], config['dmd']['shift'])
    def snapshot(step):
        g.eval()
        with torch.random.fork_rng(devices=[device]), torch.no_grad():
            for case in cases:
                rng = torch.Generator(device=device).manual_seed(42 + case)
                result, meta = student_rollout(g, fixed[case], schedule, 9, 0, rng=rng, train_mode=False)
                torch.save(result.cpu(), output / f'step{step}.case{case:03d}.pt')
        g.train()
        dist.barrier()
    baseline = Path(config['validation']['baseline_root'])
    header = dict(source_checkpoint=str(Path(config['source']['phase2_checkpoint']).resolve()),
        source_bundle_sha256=config['source']['score_bundle_sha256'], cases=config['validation']['indices'],
        sampler_sigmas=list(schedule), frames=158, seed='42+case_id', model_G=0)
    def baseline_ready():
        marker = baseline / 'BASELINE.json'
        if not marker.exists():
            return False
        if json.loads(marker.read_text()) != header:
            raise ValueError('Original fewstepG0 baseline provenance differs')
        return all((baseline / f'step0.case{case:03d}.pt').is_file() for case in header['cases'])
    if not rank0_check(baseline_ready):
        if counters.generator != 0 or config.get('continuation') or baseline.resolve() != output:
            raise ValueError('Original fewstepG0 baseline missing; refuse to relabel a resumed model')
        status('fewstep_G0_baseline', complete_G=0, complete_C=counters.critic)
        snapshot(0)
        if rank == 0:
            (baseline / 'BASELINE.json').write_text(json.dumps(header, indent=2) + '\n')
        dist.barrier()
    engine = Engine(g, teacher, c, go, co, counters, reader, schedule, config['dmd'], device, streams)
    receipts = output / 'step_receipts'
    receipts.mkdir(exist_ok=True)
    def record(value):
        role = value['role']
        step = value['critic'] if role == 'critic' else value['generator']
        prefix = 'CRITIC' if role == 'critic' else 'COMPLETE'
        path = receipts / f'{prefix}.step{step:06d}.rank{rank:03d}.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value) + '\n')
        temporary.replace(path)
        if rank == 0:
            print(json.dumps(value), flush=True)
            with (output / 'log.txt').open('a') as stream:
                stream.write(json.dumps(value) + '\n')
    while engine.counters.critic < engine.counters.critic_warmup:
        status('critic_warmup', complete_G=0, complete_C=engine.counters.critic)
        record(engine.update('critic'))
    points = set(config['validation_steps'])
    while engine.counters.generator < config['train']['max_generator_steps']:
        status('training', complete_G=engine.counters.generator, complete_C=engine.counters.critic)
        for _ in range(5):
            record(engine.update('critic'))
        record(engine.update('generator'))
        step = engine.counters.generator
        if engine.counters.critic > config['train']['max_critic_steps']:
            raise RuntimeError('Explicit critic update budget exceeded')
        if step in config['checkpoint_steps']:
            status('checkpointing', complete_G=step, complete_C=engine.counters.critic)
            save_checkpoint(output, g, c, go, co, engine.counters, config, device, rng_streams=streams)
        if step in points:
            status('saving_validation_latents', complete_G=step, complete_C=engine.counters.critic)
            snapshot(step)
    final = asdict(engine.counters)
    # Free all three roles and Adam before native VAE. Validation timing changes
    # memory placement only; all selected checkpoints/cases are still decoded.
    del engine, g, teacher, c, go, co
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    status('decoding_validation', **final)
    for step in sorted(points - {0}):
        if step <= config['train']['initial_generator_step']:
            continue
        directory = output if step == final['generator'] else output / f'validation_step{step:06d}'
        directory.mkdir(exist_ok=True)
        for case in cases:
            before = torch.load(baseline / f'step0.case{case:03d}.pt', map_location='cpu', weights_only=True)
            current = torch.load(output / f'step{step}.case{case:03d}.pt', map_location='cpu', weights_only=True)
            deliver(p2['model_base'], directory, fixed[case], before, current, case, device,
                rank=rank, step=step, baseline_step=0, model_label='BiWM H3 DMD4map',
                baseline_label=f'Initial DMD G0 from FM Live{source["step"]}')
            marker = directory / 'validation' / f'case{case:03d}' / 'COMPLETE.json'
            record = json.loads(marker.read_text())
            record.update(stage='BiWM H3 phase3 DMD', generator_step=step,
                critic_step=config['train']['critic_warmup'] + 5 * step,
                source_phase2_step=source['step'], sampler_maps=4)
            marker.write_text(json.dumps(record, indent=2) + '\n')
        dist.barrier()
    status('DMD_validation_complete', **final, source_phase2_step=source['step'],
           validation_frames=158, validation_cases=8, quality_review_pending=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
