# SPDX-License-Identifier: Apache-2.0
"""Minimal config-driven4/8-H200 FramePack adaptation, effective batch8."""
from datetime import timedelta
from functools import partial
import gc, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from h3.data import CameraDataset
from h3.distributed import clip_sharded_grad_norm
from h3.h3_fa3_policy import require_h3_fa3_training_backend
from h3.fa3_autograd import verify_fa3_backward
from h3.layout import unpatchify
from h3.optimizer import FP32MasterAdamW
from pipelines.h3.train_stage1 import configure_lora, save_checkpoint
from h3.framepack import FramePackH3, loss
from h3.framepack_data import SolarWMCameraDataset, GlobalSampleOrder
from h3.framepack_checkpoint import restore_training


def wrap(model, device):
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock, MiniMaxH3TokenRefinerBlock
    fp32 = {model.proj_in, model.proj_out, model.audio_proj_in, model.audio_proj_out,
            model.time_embedder, model.history_compressor}
    policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda module:
                     module in fp32 or isinstance(module,(MiniMaxH3TransformerBlock,MiniMaxH3TokenRefinerBlock)))
    return FSDP(model, auto_wrap_policy=policy, device_id=device, use_orig_params=True,
                sharding_strategy=ShardingStrategy.FULL_SHARD, sync_module_states=device.type=='cuda',
                limit_all_gathers=True, mixed_precision=MixedPrecision(param_dtype=None,reduce_dtype=torch.float32))


@torch.no_grad()
def rollout(model, sample, seed, latent_frames=12, points=30):
    if latent_frames < 7 or (latent_frames-7)%5 or latent_frames > sample['target_latents'].shape[1]:
        raise ValueError('Rollout must contain complete native first7/later5 latent chunks')
    if points < 2:raise ValueError('At least two sigma schedule points are required')
    generator = torch.Generator(device=sample['target_latents'].device).manual_seed(seed)
    device = sample['target_latents'].device
    def noise(shape):
        return torch.randn(shape,device=device,generator=generator,dtype=torch.float32)
    sample = dict(sample)
    sample['target_latents'] = torch.zeros_like(sample['target_latents'])
    anchor = .999*sample['anchor_latents'].float().unsqueeze(0)
    anchor += .001*noise(anchor.shape)
    sigmas=torch.linspace(1,0,points,device=device);sigmas=3*sigmas/(1+2*sigmas)
    generated=[]
    chunks=[(0,7)]+[(start,5) for start in range(7,latent_frames,5)]
    for start,count in chunks:
        x=noise((1,24,count,*sample['target_latents'].shape[-2:]))
        for sigma,nxt in zip(sigmas[:-1],sigmas[1:]):
            velocity=model(sample,x,anchor,1-sigma,start)
            x += (sigma-nxt)*unpatchify(velocity,count,*sample['target_latents'].shape[-2:])
        generated.append(x)
        sample['target_latents'][:,start:start+count]=x[0]
    result=torch.cat(generated,dim=2)
    if not torch.isfinite(result).all():raise RuntimeError('Nonfinite AR rollout')
    return result.cpu()


def deliver(base, output, sample, before, after, case_id, device, *, rank, step=10, baseline_step=0,
            model_label='FramePack', baseline_label=None):
    from diffusers import AutoencoderKLMiniMaxH3
    from PIL import Image,ImageDraw
    import subprocess
    latent_frames=before.shape[2]
    if after.shape!=before.shape:raise ValueError('Baseline/result latent geometry differs')
    frames=22+17*((latent_frames-7)//5)
    height,width=[int(size)*16 for size in sample['target_latents'].shape[-2:]]
    vae=AutoencoderKLMiniMaxH3.from_pretrained(base,subfolder='vae',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
    from h3.vae_attention import configure_vae_attention
    configure_vae_attention(vae)
    mean=torch.tensor(vae.config.latents_mean,device=device).view(1,-1,1,1,1)
    std=torch.tensor(vae.config.latents_std,device=device).view(1,-1,1,1,1)
    def decode(latent):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            rgb=vae.decode(latent.to(device).float()*std+mean,return_dict=False)[0]
        rgb=rgb.float()*torch.tensor((.229,.224,.225),device=device).view(1,3,1,1,1)
        rgb+=torch.tensor((.485,.456,.406),device=device).view(1,3,1,1,1)
        if rgb.shape!=(1,3,frames,height,width) or not torch.isfinite(rgb).all():raise RuntimeError('Native RGB decode differs')
        return (rgb.clamp(0,1)[0].permute(1,2,3,0)*255).round().byte().cpu().numpy()
    gt=decode(sample['target_latents'][:,:latent_frames].unsqueeze(0));b=decode(before);a=decode(after)
    directory=output/'validation'/f'case{case_id:03d}';directory.mkdir(parents=True,exist_ok=True)
    def encode(name,panels,labels):
        frames=np.concatenate(panels,axis=2).copy()
        for index in range(len(frames)):
            image=Image.fromarray(frames[index]);draw=ImageDraw.Draw(image)
            for j,label in enumerate(labels):
                draw.rectangle((j*width,0,(j+1)*width,28),fill='black')
                draw.text((j*width+8,6),label,fill='white')
            frames[index]=np.asarray(image)
        path=directory/name
        process=subprocess.Popen(['ffmpeg','-y','-v','error','-threads','2','-f','rawvideo','-pix_fmt','rgb24','-s',f'{frames.shape[2]}x{height}','-r','24','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
        process.communicate(frames.tobytes())
        if process.returncode:raise RuntimeError('MP4 encode failed')
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(path)]))
        if int(probe['streams'][0]['nb_read_frames'])!=len(frames):raise RuntimeError('Delivered video frame count differs')
        subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
    before_label=baseline_label or f'Baseline Live step{baseline_step} seed{42+case_id}'
    after_label=f'{model_label} Live step{step} seed{42+case_id}'
    encode('baseline.mp4',[b],[before_label])
    encode(f'step{step}.mp4',[a],[after_label])
    encode(f'GT_step{step}.mp4',[gt,a],['GT VAE reconstruction',after_label])
    encode(f'GT_baseline_step{step}.mp4',[gt,b,a],['GT VAE reconstruction',before_label,after_label])
    (directory/'COMPLETE.json').write_text(json.dumps(dict(rank=rank,case_id=case_id,frames=frames,seed=42+case_id,model_step=step,baseline_step=baseline_step,GT='VAE reconstruction; original RGB not locally extracted',full_decode=True,scope='native autoregressive rollout; quality review is separate'),indent=2)+'\n')


def validate_recipe(config):
    """Reject metadata that contradicts the fixed camera FM implementation."""
    fixed = {
        'objective': dict(shift=3, first_chunk_latents=7, later_chunk_latents=5,
                          teacher_forcing=True),
        'train': dict(gpu='H200', FSDP='FULL_SHARD', SP=1, micro_batch=1,
                      effective_batch=8, lora_targets=312, train_compressor=True,
                      attention_backend='_flash_3'),
        'optimizer': dict(name='FP32MasterAdamW', warmup=0, schedule='constant'),
        'validation': dict(CFG=1),
        'data': dict(qwen_skip_reference_videos=True),
    }
    for section, fields in fixed.items():
        for key, expected in fields.items():
            if config[section].get(key) != expected:
                raise ValueError(f'Fixed Stage2 recipe requires {section}.{key}={expected!r}')
    allowed = config['train']['allowed_world_sizes']
    if not allowed or not set(allowed) <= {4, 8}:
        raise ValueError('Stage2 supports only4/8 H200 ranks')
    points = config['validation']['schedule_points']
    if type(points) is not int or points < 2:
        raise ValueError('Validation requires at least two integer schedule points')


def main():
    output=Path(os.environ['BIWM_RUN']).resolve();config=json.loads((output/'config.json').read_text())
    validate_recipe(config)
    frames=config['validation']['rollout_latents']
    if frames<7 or (frames-7)%5 or frames>config['data']['latent_frames']:
        raise ValueError('Validation requires complete native latent chunks within the source video')
    if config['validation']['rollout_rgb_frames']!=22+17*((frames-7)//5):
        raise ValueError('Validation RGB/latent frame counts disagree')
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);device=torch.device('cuda',local)
    require_h3_fa3_training_backend('_flash_3',device)
    dist.init_process_group('nccl',timeout=timedelta(minutes=90));rank,world=dist.get_rank(),dist.get_world_size()
    fa3_receipt=dict(rank=rank,world_size=world,**verify_fa3_backward(device))
    (output/f'FA3_GATE.rank{rank:03d}.json').write_text(json.dumps(fa3_receipt,indent=2)+'\n')
    print(json.dumps(dict(event='fa3_backward_gate',**fa3_receipt)),flush=True)
    dist.barrier()
    if world not in config['train']['allowed_world_sizes']:raise ValueError('Authorized world sizes are4 or8 H200')
    if 'H200' not in torch.cuda.get_device_name(device):raise ValueError('H200 required')
    accumulation=config['train']['effective_batch']//world
    if world*accumulation!=8:raise ValueError('Effective batch must stay8')
    maximum=int(config['train']['max_steps'])
    if maximum<1:raise ValueError('A positive explicit training budget is required')
    config['train'].update(world_size=world,DP=world,accumulation=accumulation)
    if rank==0:(output/'executed_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    def status(phase,**extra):
        if rank==0:(output/'STATUS.json').write_text(json.dumps(dict(phase=phase,at=time.time(),**extra),indent=2)+'\n')
    seed=config['train']['seed']
    torch.manual_seed(seed+rank);random.seed(seed+rank);np.random.seed(seed+rank)
    initialization=Path(config['initialization_path']);data=Path(config['data_path'])
    merge=json.loads((initialization/'MERGE_COMPLETE.json').read_text())
    if merge['modules']!=312:raise ValueError('Camera merge incomplete')
    status('loading_merged_camera_model')
    model=FramePackH3.strict_from_pretrained(initialization,subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
    model.install_framepack();model.set_attention_backend('_flash_3')
    from argparse import Namespace
    configure_lora(model,Namespace(lora_rank=config['train']['lora_rank'],lora_alpha=config['train']['lora_alpha']))
    model.history_compressor.requires_grad_(True);model.enable_gradient_checkpointing()
    model=wrap(model,device)
    opt=config['optimizer']
    optimizer=FP32MasterAdamW((p for p in model.parameters() if p.requires_grad),lr=opt['lr'],betas=tuple(opt['betas']),eps=opt['eps'],weight_decay=opt['weight_decay'])
    if config['data'].get('train_manifest'):
        manifest=Path(config['data']['train_manifest'])
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=config['data']['train_manifest_sha256']:
            raise ValueError('Full training manifest hash differs')
        train=SolarWMCameraDataset(manifest,config['data']['release_root'])
        order=GlobalSampleOrder(len(train),seed)
    else:
        train=CameraDataset(data/'train.jsonl');order=None
    validation=CameraDataset(data/'validation.jsonl')
    all_cases=config['validation'].get('indices',list(range(config['validation']['cases'])))
    if len(all_cases)!=config['validation']['cases'] or len(set(all_cases))!=len(all_cases):raise ValueError('Validation cases must be fixed unique indices')
    case_ids=all_cases[rank::world]
    if config['validation']['cases']%world:raise ValueError('All FSDP ranks need the same validation wave count')
    fixed={case:{k:v.to(device) for k,v in validation[case].items()} for case in case_ids}
    step,local_resume=restore_training(output,model,optimizer,config,rank,world,device)
    if step>maximum:raise ValueError('Checkpoint step exceeds configured budget')
    if not local_resume and config.get('continuation') and maximum<=step:
        raise ValueError('Continuation must explicitly budget updates after the source checkpoint')
    sample_origin=config['train'].get('sample_step_origin',0)
    if sample_origin>step:raise ValueError('Data sample origin is ahead of optimizer step')
    latent_frames=config['validation']['rollout_latents'];points=config['validation']['schedule_points']
    baseline_manifest=output/'BASELINE.json'
    if local_resume and not baseline_manifest.exists():raise RuntimeError('Baseline provenance missing on resume')
    baseline_info=json.loads(baseline_manifest.read_text()) if baseline_manifest.exists() else None
    if baseline_info and (baseline_info['cases']!=all_cases or baseline_info['latent_frames']!=latent_frames or baseline_info['schedule_points']!=points):
        raise ValueError('Saved baseline configuration differs')
    baseline_step=baseline_info['model_step'] if baseline_info else step
    if baseline_step>step:raise ValueError('Baseline is newer than the restored model')
    before={}
    for case in case_ids:
        baseline_path=output/f'baseline.case{case:03d}.pt'
        present=torch.tensor(int(baseline_path.exists()),device=device);dist.all_reduce(present,op=dist.ReduceOp.MIN)
        if not present.item():
            if local_resume:raise RuntimeError('Saved baseline missing on resume; refuse to relabel the restored model')
            status('baseline_short_rollout',step=step,case_wave=case//world);model.eval()
            torch.save(rollout(model,fixed[case],42+case,latent_frames,points),baseline_path)
        before[case]=torch.load(baseline_path,map_location='cpu',weights_only=True)
    dist.barrier()
    if rank==0 and not baseline_manifest.exists():
        baseline_manifest.write_text(json.dumps(dict(model_step=baseline_step,cases=all_cases,latent_frames=latent_frames,schedule_points=points,seed='42+case_id'),indent=2)+'\n')
    dist.barrier()
    model.train();(output/'step_receipts').mkdir(exist_ok=True)
    while step<maximum:
        status('training',complete_step=step);started=time.monotonic();optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        value_sum=torch.zeros((),device=device)
        sample_ids=[]
        for micro in range(accumulation):
            absolute=(step-sample_origin)*8+micro*world+rank
            index=order(absolute) if order is not None else absolute%len(train)
            sample_ids.append(train.rows[index].get('sample_id'))
            sample={k:v.to(device) for k,v in train[index].items()}
            value=loss(model,sample);finite=torch.isfinite(value.detach()).int();dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if not finite.item():raise RuntimeError('Nonfinite loss')
            (value/accumulation).backward();value_sum+=value.detach()/accumulation
        norm=clip_sharded_grad_norm(model.parameters(),opt['clip'])
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite global gradient norm')
        optimizer.step();step+=1;mean=value_sum.clone();dist.all_reduce(mean)
        receipt=dict(step=step,rank=rank,world_size=world,accumulation=accumulation,effective_batch=8,sample_ids=sample_ids,loss=float(value_sum),global_loss=float(mean/world),gradient_norm=float(norm),seconds=time.monotonic()-started,peak_GiB=torch.cuda.max_memory_allocated(device)/2**30)
        (output/'step_receipts'/f'COMPLETE.step{step:06d}.rank{rank:03d}.json').write_text(json.dumps(receipt)+'\n')
        if rank==0:
            print(json.dumps(receipt),flush=True)
            with (output/'log.txt').open('a') as stream:stream.write(json.dumps(receipt)+'\n')
        if step in config['train']['checkpoint_steps'] or step==maximum:save_checkpoint(output,model,optimizer,step,config,rank,world,device)
        if step in config['validation'].get('checkpoint_steps',[]) and step<maximum:
            status('intermediate_validation',complete_step=step);model.eval()
            python_state=random.getstate();numpy_state=np.random.get_state()
            with torch.random.fork_rng(devices=[device]):
                snapshot=output/f'validation_step{step:06d}';snapshot.mkdir(exist_ok=True)
                for case in case_ids:
                    current=rollout(model,fixed[case],42+case,latent_frames,points)
                    torch.save(current,snapshot/f'step{step}.case{case:03d}.pt')
                    deliver(config['model_base'],snapshot,fixed[case],before[case],current,case,device,rank=rank,step=step,baseline_step=baseline_step)
                dist.barrier()
            random.setstate(python_state);np.random.set_state(numpy_state)
            gc.collect();torch.cuda.empty_cache();model.train()
    status('final_short_rollout',complete_step=step);model.eval()
    after={}
    for case in case_ids:
        after[case]=rollout(model,fixed[case],42+case,latent_frames,points);torch.save(after[case],output/f'step{step}.case{case:03d}.pt')
    del optimizer,model;gc.collect();torch.cuda.empty_cache();dist.barrier()
    status('decoding_validation',complete_step=step)
    for case in case_ids:deliver(config['model_base'],output,fixed[case],before[case],after[case],case,device,rank=rank,step=step,baseline_step=baseline_step)
    dist.barrier();status('short_validation_complete',complete_step=step,long_training_started=bool(config['data'].get('train_manifest')),DMD_started=False)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
