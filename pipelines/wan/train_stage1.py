#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# isort: skip_file
"""BiWM Stage 1 — camera-control fine-tuning (Wan training entry).

Wan 2.2 5B training: video generation, discrete/continuous camera control,
FSDP, bf16 mixed precision, Context Parallel, multi-resolution, and partial
fine-tuning (trainable_modules).

Usage:
    torchrun --nproc_per_node 8 pipelines/wan/train_stage1.py \
        --pretrained_model_path Wan-AI/Wan2.2-TI2V-5B \
        --output_dir ./outputs_wan_22
"""

import argparse
import datetime
import gc
import json
import os
import random
import sys
import time
from typing import Any, List, Optional, Tuple

# Set NCCL timeout-related environment variables (must be before importing torch)
os.environ.setdefault("NCCL_TIMEOUT", "7200000")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from wan.modules.model import WanModel, WanAttentionBlock

from pipelines.dataset.biwm_camera_text_dataset import BiwmCamCaptionData, biwm_collate
from pipelines.wan import wan_common as wc
from pipelines.wan.wan_common import (
    LowRankLinear, _swap_in_lora, inject_lora_into_wan, toggle_lora_active, register_lora_args,
    WanVAEAdapter, _read_sharded_safetensors, _resolve_vae_pth, build_optimizer_and_scheduler,
    build_train_loader, echo_on_main_rank, embed_text_node_shared, enclose_wan_model_with_fsdp,
    free_gpu_memory, init_wan_text_encoder, init_wan_transformer, init_wan_vae, parse_step_from_checkpoint,
    read_checkpoint_for_resume, setup_wan_model, stitch_videos_with_labels, store_video, wan_vae_compress,
    wan_vae_reconstruct, CheckpointableDistSampler, fit_frames_for_wan, resolve_wan_version,
    WAN_VAE_TIME_STRIDE, WAN_TEXT_LENGTH, WAN_PATCH_DIM, WAN_NEG_PROMPT_TEXT, WAN_TI2V_5B_SETTINGS, WAN_LATENT_AVG, WAN_LATENT_STDDEV, WAN21_LATENT_CHANNEL_COUNT, WAN21_VAE_CHANNEL_DIM, WAN21_VAE_SPACE_STRIDE, WAN21_LATENT_AVG, WAN21_LATENT_STDDEV, CFG_GUIDANCE_WEIGHT, DEFAULT_VIDEO_AREA_CAP, DIFFUSION_STEP_COUNT, EVAL_INTERVAL,
)

from pipelines.utils.checkpoint import store_checkpoint
from pipelines.utils.parallel_states import (
    teardown_sequence_parallel_group,
    fetch_sequence_parallel_state,
    setup_sequence_parallel_state,
)
from pipelines.utils.context_parallel import (
    setup_context_parallel,
    teardown_context_parallel,
    cp_is_active,
    fetch_cp_settings,
    fetch_cp_world_size,
)


_DATALOADER_END = object()  # sentinel for dataloader exhaustion

def perform_single_training_step(
    transformer: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    vae_encoder: nn.Module,
    text_encoder: nn.Module,
    encode_fn: callable,
    gradient_accumulation_steps: int,
    device: torch.device,
    args: argparse.Namespace,
    current_step: int,
    distributed_rank: int,
    world_size: int,
    max_gradient_norm: float,
    data_item_override: Any = None,
    dataloader_iter: Any = None,   # real gradient accumulation — each micro-step pulls a different batch from this
    vae_decoder: nn.Module = None,
) -> Tuple:
    """
    Execute a single training step (Wan 2.2 version with advanced pipeline features).

    Returns:
        (loss, grad_norm, prompt, camera_kwargs, task_type, cond_end,
         cond_latent, gt_latent, source, caption_type, K_ctrl, c2w_ctrl,
         K_ctrl_raw, c2w_ctrl_raw, video_id, sigma)
    """
    accumulated_loss = 0.0
    gradient_norm = torch.tensor(0.0)
    last_prompt = ""
    camera_kwargs = {}
    current_task = getattr(args, 'training_mode', 't2v')
    cond_end = 0
    last_cond_latent = None
    last_gt_video_latent = None
    last_gt_pixel_frames = None  # Store raw pixel frames for GT video (avoid VAE decode issues)
    is_camera_batch = False
    optimizer.zero_grad()

    is_main_process = (distributed_rank == 0)
    def log_main(msg: str):
        if is_main_process:
            print(msg, flush=True)

    current_rank = dist.get_rank()

    for accumulation_idx in range(gradient_accumulation_steps):
        # Sync random seed
        if current_rank == 0:
            random_seed = torch.tensor([random.random()], dtype=torch.float32, device=device)
        else:
            random_seed = torch.ones(1, dtype=torch.float32, device=device)
        dist.broadcast(random_seed, src=0)

        # real gradient accumulation: each micro-step pulls a different batch
        # (micro-step 0 reuses the caller's override; the rest pull from dataloader_iter, falling back to override at epoch end).
        if accumulation_idx == 0 or dataloader_iter is None:
            data_item = data_item_override
        else:
            data_item = next(dataloader_iter, data_item_override)

        # Detect skip data (dataset returns {"skip": True} on read failure)
        _is_skip = isinstance(data_item, dict) and bool(data_item.get("skip"))
        local_skip = torch.tensor([1.0 if _is_skip else 0.0], device=device)
        dist.all_reduce(local_skip, op=dist.ReduceOp.MAX)
        if local_skip.item() > 0.5:
            log_main(f"[Rank {current_rank}] Skip data detected, global sync skip")
            return (0.0, 0.0, last_prompt, camera_kwargs, current_task, cond_end,
                    None, None, "", "", None, None, None, None, "", 0.0)

        # Read the unified sample dict (collate_fn=biwm_collate -> one dict, no batch dim)
        video_pixels = data_item["pixel_values"]        # [T,C,H,W]
        text_caption = data_item["caption"]
        video_id = data_item["video_id"]
        has_camera = data_item["has_camera"]
        source = data_item["source"]
        caption_type = data_item["caption_type"]
        precomp_action_labels = data_item["action_labels"]   # [t_lat] discrete labels
        # continuous-camera fields removed; None placeholders kept for return-tuple compatibility
        K_ctrl = c2w_ctrl = K_ctrl_raw = c2w_ctrl_raw = None
        pose_orig_w, pose_orig_h = 0.0, 0.0
        is_camera_batch = bool(has_camera)

        total_frames = video_pixels.shape[0]

        # Frame alignment (Wan VAE: (F-1) % 4 == 0)
        adjusted_frames = fit_frames_for_wan(total_frames, verbose=False)

        # Sync frames across ranks
        frames_tensor = torch.tensor([adjusted_frames], device=device, dtype=torch.long)
        dist.broadcast(frames_tensor, src=0)
        synced_frames = int(frames_tensor.item())

        # CP alignment
        if cp_is_active():
            # For Wan: compute divisible frames with vae_temporal_factor=4
            latent_frames = (synced_frames - 1) // WAN_VAE_TIME_STRIDE + 1
            cp_ws = fetch_cp_world_size()
            if latent_frames % cp_ws != 0:
                latent_frames = latent_frames - (latent_frames % cp_ws)
                synced_frames = (latent_frames - 1) * WAN_VAE_TIME_STRIDE + 1
            log_main(f"[CP] Aligned: {synced_frames} frames (cp_size={cp_ws})")

        if synced_frames != total_frames:
            video_pixels = video_pixels[:synced_frames]

        # === Caption processing ===
        if isinstance(text_caption, (list, tuple)):
            prompt_text = text_caption[0]
            while isinstance(prompt_text, (list, tuple)) and len(prompt_text) > 0:
                prompt_text = prompt_text[0]
        else:
            prompt_text = text_caption
        if not isinstance(prompt_text, str):
            prompt_text = str(prompt_text)
        last_prompt = prompt_text

        # === Text encoding ===
        _node_group = getattr(args, 'node_group', None)
        if _node_group is not None:
            context = embed_text_node_shared(
                text_encoder, encode_fn, prompt_text,
                getattr(args, 'local_rank', 0),
                getattr(args, 'num_local_ranks', 1),
                _node_group,
                getattr(args, 'node_leader_rank', 0),
                device,
                offload_te=getattr(args, 'offload_text_encoder', False),
            )
        else:
            results = encode_fn([prompt_text])
            context = results[0].to(device, dtype=torch.bfloat16)

        # === VAE encoding ===
        with torch.no_grad():
            video_pix = video_pixels.permute(1, 0, 2, 3).contiguous().to(device).float()
            C, T, H, W = video_pix.shape

            # Multi-resolution training
            _high_res_step = getattr(args, 'high_res_step', 0)
            _high_res_prob = getattr(args, 'high_res_prob', 0.0)
            if _high_res_step > 0 and current_step >= _high_res_step:
                _high_res_prob = 1.0
            if _high_res_prob > 0 and _high_res_prob < 1.0:
                _res_flag = torch.tensor([0.0], device=device)
                if current_rank == 0:
                    _res_flag[0] = 1.0 if random.random() < _high_res_prob else 0.0
                dist.broadcast(_res_flag, src=0)
                _use_high_res = _res_flag[0].item() > 0.5
            elif _high_res_prob >= 1.0:
                _use_high_res = True
            else:
                _use_high_res = False

            _target_h = getattr(args, 'high_res_height', H) if _use_high_res else args.num_height
            _target_w = getattr(args, 'high_res_width', W) if _use_high_res else args.num_width
            if _target_h != H or _target_w != W:
                video_pix = torch.nn.functional.interpolate(
                    video_pix, size=(_target_h, _target_w),
                    mode='bicubic', align_corners=False, antialias=True,
                )
                C, T, H, W = video_pix.shape
                log_main(f"[MultiRes] {'720p' if _use_high_res else '540p'}: resize -> {H}x{W}")

            # Dataset already normalizes to [-1, 1]; clamp since bicubic resize can overshoot.
            video_pix = video_pix.clamp(-1.0, 1.0)

            _pixel_H, _pixel_W = H, W

            # Ensure (T-1) % 4 == 0
            if ((T - 1) % WAN_VAE_TIME_STRIDE) != 0:
                x = (T - 1) // WAN_VAE_TIME_STRIDE
                valid_T = 1 + WAN_VAE_TIME_STRIDE * x
                video_pix = video_pix[:, :valid_T, :, :]
                T = valid_T

            # Save raw pixels for GT before encode
            _gt_pix = ((video_pix.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu()
            last_gt_pixel_frames = [_gt_pix[i].numpy() for i in range(_gt_pix.shape[0])]

            video_latent = wan_vae_compress(vae_encoder, video_pix, device)
            del video_pix

        # CP sync
        if cp_is_active():
            cp_cfg = fetch_cp_settings()
            cp_leader = dist.get_rank() - cp_cfg.cp_rank
            dist.broadcast(video_latent, src=cp_leader, group=cp_cfg.cp_group)
            dist.broadcast(context, src=cp_leader, group=cp_cfg.cp_group)

        C_lat, T_lat, H_lat, W_lat = video_latent.shape

        # === Noise & Sigma ===
        noise = torch.randn_like(video_latent)
        sigma = torch.rand(1, device=device, dtype=video_latent.dtype)
        # Optional: apply shift to sigma (bias toward the high-noise region, aligned with the inference schedule)
        _sigma_shift = getattr(args, 'train_sigma_shift', 0.0)
        if _sigma_shift > 0:
            sigma = _sigma_shift * sigma / (1 + (_sigma_shift - 1) * sigma)

        if cp_is_active():
            dist.broadcast(noise, src=cp_leader, group=cp_cfg.cp_group)
            dist.broadcast(sigma, src=cp_leader, group=cp_cfg.cp_group)

        # === Task type (t2v/i2v/v2v/hybrid) ===
        training_mode = getattr(args, 'training_mode', 't2v')
        if training_mode == "hybrid":
            i2v_prob = getattr(args, 'i2v_prob', 0.3)
            v2v_prob = getattr(args, 'v2v_prob', 0.2)
            if current_rank == 0:
                task_rand = random.random()
                if task_rand < i2v_prob:
                    task_choice_val = torch.tensor([1.0], device=device)
                elif task_rand < i2v_prob + v2v_prob:
                    task_choice_val = torch.tensor([2.0], device=device)
                else:
                    task_choice_val = torch.tensor([0.0], device=device)
            else:
                task_choice_val = torch.zeros(1, device=device)
            dist.broadcast(task_choice_val, src=0)
            current_task = ["t2v", "i2v", "v2v"][int(task_choice_val.item())]
        else:
            current_task = training_mode

        cond_end = 0
        if current_task == "i2v":
            cond_end = min(getattr(args, 'i2v_cond_latent_frames', 1), T_lat - 1)
        elif current_task == "v2v":
            v2v_cond = getattr(args, 'v2v_cond_latent_frames', 0)
            if v2v_cond > 0:
                cond_end = min(v2v_cond, T_lat - 1)
            else:
                # Random v2v conditioning ratio (broadcast from rank 0)
                _v2v_min = getattr(args, 'v2v_cond_ratio_min', None)
                _v2v_max = getattr(args, 'v2v_cond_ratio_max', None)
                if _v2v_min is not None and _v2v_max is not None and _v2v_min < _v2v_max:
                    _ratio_t = torch.zeros(1, device=device)
                    if current_rank == 0:
                        _ratio_t[0] = random.uniform(_v2v_min, _v2v_max)
                    dist.broadcast(_ratio_t, src=0)
                    cond_end = max(1, int(T_lat * _ratio_t.item()))
                else:
                    cond_end = max(1, int(T_lat * getattr(args, 'v2v_cond_ratio', 0.25)))
                cond_end = min(cond_end, T_lat - 1)

        # === validation_only: skip forward/backward ===
        if getattr(args, 'validation_only', False):
            log_main(f"[validation_only] Skip training, keep data for validation")
            last_cond_latent = video_latent[:, :cond_end].detach().clone() if cond_end > 0 else None
            last_gt_video_latent = video_latent.detach().clone()
            _vid_id = str(video_id) if video_id is not None else "unknown"
            return (0.0, 0.0, last_prompt, camera_kwargs, current_task, cond_end,
                    last_cond_latent, last_gt_video_latent,
                    source, caption_type, K_ctrl, c2w_ctrl, K_ctrl_raw, c2w_ctrl_raw,
                    _vid_id, 0.0, last_gt_pixel_frames)

        # === Add noise ===
        sigma_expanded = sigma.view(1, 1, 1, 1)
        if cond_end > 0:
            cond_mask = torch.zeros(1, T_lat, 1, 1, device=device, dtype=video_latent.dtype)
            cond_mask[:, :cond_end] = 1.0
            noisy_latent = cond_mask * video_latent + (1 - cond_mask) * (
                (1 - sigma_expanded) * video_latent + sigma_expanded * noise
            )
        else:
            noisy_latent = (1 - sigma_expanded) * video_latent + sigma_expanded * noise

        # === Camera control ===
        camera_kwargs = {}
        _use_cam = getattr(args, 'use_camera_control', False)

        # === BiWM: use the discrete action_labels precomputed from action_frames (latent-level, no continuous Plucker) ===
        if _use_cam and precomp_action_labels is not None:
            _al = precomp_action_labels
            if not isinstance(_al, torch.Tensor):
                _al = torch.as_tensor(_al)
            if _al.dim() == 1:
                _al = _al.unsqueeze(0)              # [1, L]
            _al = _al.to(device=device, dtype=torch.long)
            # Align to the current latent frame count T_lat (when frames are truncated/CP-aligned), using nearest resampling
            if _al.shape[1] != T_lat:
                _al = F.interpolate(_al.float().unsqueeze(1), size=T_lat,
                                    mode='nearest').squeeze(1).long()
            camera_kwargs['action_labels'] = _al

        # CP sync camera_kwargs
        if cp_is_active():
            _cam_flag = torch.tensor([1.0 if camera_kwargs else 0.0], device=device)
            dist.broadcast(_cam_flag, src=cp_leader, group=cp_cfg.cp_group)
            if _cam_flag.item() <= 0.5:
                camera_kwargs = {}

        dist.barrier()

        # === Forward (Wan-style) ===
        seq_len = T_lat * (H_lat // WAN_PATCH_DIM[1]) * (W_lat // WAN_PATCH_DIM[2])
        x_input = [noisy_latent]
        t_input = (sigma * 1000.0).expand(1).to(device)
        context_input = [context]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            video_output_list = transformer(
                x=x_input,
                t=t_input,
                context=context_input,
                seq_len=seq_len,
                cond_latent_frames=cond_end,
                **camera_kwargs,
            )
            # wan.modules.model returns List[Tensor], stack to [B, C, F, H, W]
            if isinstance(video_output_list, (list, tuple)):
                video_output = torch.stack(video_output_list)
            else:
                video_output = video_output_list

        # === Loss (velocity matching) ===
        if cond_end > 0:
            velocity_target = (noise - video_latent)[:, cond_end:]
            velocity_target = velocity_target.unsqueeze(0)
            output_for_loss = video_output[:, :, cond_end:]
            loss = F.mse_loss(output_for_loss.float(), velocity_target.float())
        else:
            velocity_target = (noise - video_latent).unsqueeze(0)
            loss = F.mse_loss(video_output.float(), velocity_target.float())

        # === NaN check (global sync) ===
        local_nan = torch.tensor(
            [1.0 if (torch.isnan(loss) or torch.isinf(loss)) else 0.0],
            device=device,
        )
        dist.all_reduce(local_nan, op=dist.ReduceOp.MAX)
        if local_nan.item() > 0.5:
            log_main(f"[WARNING] NaN/Inf loss detected, skip micro-step")
            continue   # keep accumulated grads (do not zero_grad during accumulation)

        # Loss clipping
        _loss_clip_thresh = 2.0
        _loss_val = loss.detach().item()
        local_loss_too_large = torch.tensor(
            [1.0 if _loss_val > _loss_clip_thresh else 0.0], device=device,
        )
        dist.all_reduce(local_loss_too_large, op=dist.ReduceOp.MAX)
        if local_loss_too_large.item() > 0.5:
            log_main(f"[WARNING] loss={_loss_val:.4f} > {_loss_clip_thresh}, skip step")
            _anomaly_log = os.path.join(getattr(args, 'output_dir', './outputs'), 'anomaly_samples.txt')
            with open(_anomaly_log, 'a', encoding='utf-8') as _af:
                _af.write(f"step={current_step}, loss={_loss_val:.4f}, "
                          f"video_id={video_id}, source={source}\n")
            continue   # keep accumulated grads (skip this micro-step)

        # scale loss by accum steps then backward; step/scheduler/zero_grad only on the last micro-step
        (loss / gradient_accumulation_steps).backward()

        accumulated_loss += loss.detach().item() / gradient_accumulation_steps

        # Update parameters only on the last micro-step (other steps only accumulate gradients)
        if accumulation_idx == gradient_accumulation_steps - 1:
            _clip_params = [p for p in transformer.parameters() if p.requires_grad]
            gradient_norm = torch.nn.utils.clip_grad_norm_(_clip_params, max_gradient_norm)

            # Gradient NaN check
            local_grad_nan = torch.tensor(
                [1.0 if (torch.isnan(gradient_norm) or torch.isinf(gradient_norm)) else 0.0],
                device=device,
            )
            dist.all_reduce(local_grad_nan, op=dist.ReduceOp.MAX)
            if local_grad_nan.item() > 0.5:
                log_main(f"[WARNING] NaN/Inf gradient, skip step")
                optimizer.zero_grad()
                continue

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # Save cond/GT latent
        if cond_end > 0:
            last_cond_latent = video_latent[:, :cond_end].detach().clone()
        else:
            last_cond_latent = None
        last_gt_video_latent = video_latent.detach().clone()

        del video_output, loss, noisy_latent, noise, video_latent, context

    # Return
    last_video_id = video_id
    if isinstance(last_video_id, (list, tuple)):
        last_video_id = last_video_id[0]
    last_video_id = str(last_video_id) if last_video_id is not None else "unknown"
    last_sigma = sigma.item() if isinstance(sigma, torch.Tensor) else float(sigma)

    return (accumulated_loss,
            gradient_norm.item() if isinstance(gradient_norm, torch.Tensor) else gradient_norm,
            last_prompt, camera_kwargs, current_task, cond_end, last_cond_latent, last_gt_video_latent,
            source, caption_type, K_ctrl, c2w_ctrl, K_ctrl_raw, c2w_ctrl_raw, last_video_id,
            last_sigma, last_gt_pixel_frames)


# =============================================================================
# Validation sampling
# =============================================================================
def perform_validation_sampling(
    transformer: nn.Module,
    vae_decoder: nn.Module,
    text_encoder: nn.Module,
    encode_fn: callable,
    prompt: str,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
    output_dir: str,
    step: int,
    distributed_rank: int,
    cfg_scale: float = 5.0,
    is_fsdp_model: bool = True,
    camera_kwargs: Optional[dict] = None,
    diffusion_sampling_steps: int = 50,
    task_type: str = "t2v",
    cond_latent: Optional[torch.Tensor] = None,
    cond_latent_frames: int = 0,
    gt_video_latent: Optional[torch.Tensor] = None,
    video_id: str = "unknown",
    fps: float = 16.0,
    source: str = "unknown",
    gt_pixel_frames=None,
) -> None:
    dist.barrier()
    was_training = transformer.training

    transformer.eval()

    # Disable CP during validation
    from pipelines.utils.context_parallel import _cp_settings
    cp_was_enabled = _cp_settings.enabled
    _cp_settings.enabled = False

    if distributed_rank == 0 or is_fsdp_model:
        with torch.no_grad():
            _perform_validation_inner(
                transformer=transformer,
                vae_decoder=vae_decoder,
                text_encoder=text_encoder,
                encode_fn=encode_fn,
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                device=device,
                output_dir=output_dir,
                step=step,
                distributed_rank=distributed_rank,
                cfg_scale=cfg_scale,
                camera_kwargs=camera_kwargs,
                diffusion_sampling_steps=diffusion_sampling_steps,
                task_type=task_type,
                cond_latent=cond_latent,
                cond_latent_frames=cond_latent_frames,
                gt_video_latent=gt_video_latent,
                video_id=video_id,
                fps=fps,
                gt_pixel_frames=gt_pixel_frames,
                source=source,
            )

    _cp_settings.enabled = cp_was_enabled
    if was_training:
        transformer.train()
    dist.barrier()


def _perform_validation_inner(
    transformer,
    vae_decoder,
    text_encoder,
    encode_fn,
    prompt,
    num_frames,
    height,
    width,
    device,
    output_dir,
    step,
    distributed_rank,
    cfg_scale=5.0,
    camera_kwargs=None,
    diffusion_sampling_steps=50,
    task_type="t2v",
    cond_latent=None,
    cond_latent_frames=0,
    gt_video_latent=None,
    video_id="unknown",
    fps=16.0,
    source="unknown",
    gt_pixel_frames=None,
    **kwargs,
):
    on_main_rank = (distributed_rank == 0)

    # Compute latent dims
    num_frames_adj = fit_frames_for_wan(num_frames)
    T_lat = (num_frames_adj - 1) // WAN_VAE_TIME_STRIDE + 1
    H_lat = height // wc.WAN_VAE_SPACE_STRIDE
    W_lat = width // wc.WAN_VAE_SPACE_STRIDE

    if on_main_rank:
        print(f"[Validation] step={step}, frames={num_frames_adj}, latent=({T_lat},{H_lat},{W_lat})")

    # Text encoding (all ranks need to participate for FSDP)
    world_size = dist.get_world_size()

    def _encode_all_prompts(prompt_text):
        prompt_bytes = prompt_text.encode('utf-8')
        len_tensor = torch.tensor([len(prompt_bytes)], device=device, dtype=torch.long)
        all_lens = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
        dist.all_gather(all_lens, len_tensor)
        max_len = max(l.item() for l in all_lens)
        padded = torch.zeros(int(max_len), device=device, dtype=torch.uint8)
        padded[:len(prompt_bytes)] = torch.tensor(list(prompt_bytes), device=device, dtype=torch.uint8)
        all_padded = [torch.zeros(int(max_len), device=device, dtype=torch.uint8) for _ in range(world_size)]
        dist.all_gather(all_padded, padded)

        if distributed_rank == 0:
            all_prompts = []
            for i in range(world_size):
                p_len = int(all_lens[i].item())
                p_bytes = bytes(all_padded[i][:p_len].cpu().tolist())
                all_prompts.append(p_bytes.decode('utf-8'))
            all_contexts = []
            for p in all_prompts:
                ctx = encode_fn([p])
                all_contexts.append(ctx[0].to(dtype=torch.bfloat16, device=device))
            max_seq = max(c.shape[0] for c in all_contexts)
            dim = all_contexts[0].shape[-1]
            shape_t = torch.tensor([max_seq, dim], device=device, dtype=torch.long)
            dist.broadcast(shape_t, src=0)
            stacked = torch.zeros(world_size, max_seq, dim, device=device, dtype=torch.bfloat16)
            for i, c in enumerate(all_contexts):
                stacked[i, :c.shape[0]] = c
            dist.broadcast(stacked, src=0)
            return stacked[distributed_rank]
        else:
            shape_t = torch.zeros(2, device=device, dtype=torch.long)
            dist.broadcast(shape_t, src=0)
            stacked = torch.zeros(world_size, int(shape_t[0].item()), int(shape_t[1].item()),
                                  device=device, dtype=torch.bfloat16)
            dist.broadcast(stacked, src=0)
            return stacked[distributed_rank]

    pos_context = _encode_all_prompts(prompt)
    neg_context = _encode_all_prompts(WAN_NEG_PROMPT_TEXT)

    seq_len = T_lat * (H_lat // WAN_PATCH_DIM[1]) * (W_lat // WAN_PATCH_DIM[2])

    # Init latent
    rank_seed = 42 + distributed_rank
    noise_gen = torch.Generator(device=device).manual_seed(rank_seed)
    video_latent = torch.randn(
        1, wc.WAN_LATENT_CHANNEL_COUNT, T_lat, H_lat, W_lat,
        device=device, dtype=torch.bfloat16, generator=noise_gen,
    )

    # Conditioning frames (I2V/V2V: conditioning frames sigma=0)
    val_cond_latent = None
    val_cond_frames = 0
    if cond_latent is not None and cond_latent_frames > 0:
        val_cond_latent = cond_latent[:, :cond_latent_frames].to(device, dtype=torch.bfloat16)
        val_cond_frames = cond_latent_frames
        video_latent[0, :, :val_cond_frames] = val_cond_latent
        if on_main_rank:
            print(f"[Validation] I2V: cond_frames={val_cond_frames}/{T_lat}, cond_t=0.0 (clean), gen_frames={T_lat - val_cond_frames}")

    # Sigma schedule (shift=5 for Wan2.2 TI2V-5B, official config)
    shift = 5.0
    sigmas = torch.linspace(1.0, 0.0, diffusion_sampling_steps + 1, device=device)
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

    val_camera_kwargs = camera_kwargs or {}

    # Denoising loop (Euler method for flow matching)
    for i in range(len(sigmas) - 1):
        sigma_val = sigmas[i].item()
        sigma_next = sigmas[i + 1].item()

        x_input = [video_latent.squeeze(0)]  # List[Tensor[C, F, H, W]]
        t_input = torch.tensor([sigma_val * 1000.0], device=device)  # timestep ∈ [0, 1000]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Positive (conditional) prediction
            pos_out_list = transformer(
                x=x_input,
                t=t_input,
                context=[pos_context],
                seq_len=seq_len,
                cond_latent_frames=val_cond_frames,
                **val_camera_kwargs,
            )
            pos_out = torch.stack(pos_out_list) if isinstance(pos_out_list, (list, tuple)) else pos_out_list

            # CFG: noise_pred = uncond + scale * (cond - uncond)
            if cfg_scale > 1.0:
                neg_out_list = transformer(
                    x=x_input,
                    t=t_input,
                    context=[neg_context],
                    seq_len=seq_len,
                    cond_latent_frames=val_cond_frames,
                )
                neg_out = torch.stack(neg_out_list) if isinstance(neg_out_list, (list, tuple)) else neg_out_list
                video_velocity = neg_out + cfg_scale * (pos_out - neg_out)
            else:
                video_velocity = pos_out

        # Euler step: x_{t-dt} = x_t + v * dt (dt < 0 since sigma decreases)
        dt = sigma_next - sigma_val
        video_latent = (video_latent.float() + video_velocity.float() * dt).to(video_latent.dtype)

        # Re-pin conditioning frames for I2V
        if val_cond_latent is not None:
            video_latent[0, :, :val_cond_frames] = val_cond_latent

    # Sanity-check final latent range (after normalization should be mean~0, std~1)
    if on_main_rank:
        _final_mean = video_latent.float().mean().item()
        _final_std = video_latent.float().std().item()
        if abs(_final_mean) > 5.0 or _final_std > 10.0 or _final_std < 0.01:
            print(f"[Validation] WARNING: Latent stats look abnormal! Expected mean~0, std~1")

    # Decode and save
    if on_main_rank or True:  # All ranks decode for FSDP
        os.makedirs(output_dir, exist_ok=True)
        _safe_vid = str(video_id).replace("/", "_").replace("\\", "_").replace(" ", "_")[:80]
        _step_str = "pretrain" if step < 0 else f"{step:06d}"
        _safe_source = str(source).replace("/", "_").replace(" ", "_")[:30]
        _has_cam_tag = "cam" if (camera_kwargs and 'action_labels' in camera_kwargs) else "nocam"
        _name_prefix = f"step_{_step_str}_{task_type}_{_has_cam_tag}_{_safe_source}_rank_{distributed_rank:03d}_{_safe_vid}"

        # Decode generated video
        frames = wan_vae_reconstruct(vae_decoder, video_latent.squeeze(0), device)

        # GT: decode gt_video_latent through VAE to see VAE reconstruction quality
        gt_frames = None
        if gt_video_latent is not None:
            _gt_lat = gt_video_latent.squeeze(0) if gt_video_latent.dim() == 5 else gt_video_latent
            gt_frames = wan_vae_reconstruct(vae_decoder, _gt_lat, device)
        elif gt_pixel_frames is not None:
            gt_frames = gt_pixel_frames
        if gt_frames is not None and len(gt_frames) != len(frames):
            indices = np.linspace(0, len(gt_frames)-1, len(frames)).astype(int)
            gt_frames = [gt_frames[i] for i in indices]

        output_path = os.path.join(output_dir, f"validation_{_name_prefix}.mp4")
        store_video(frames, output_path, fps=fps)
        print(f"[Rank {distributed_rank}] --> Validation video saved: {output_path}")

        # Save validation metadata
        _meta_path = os.path.join(output_dir, f"validation_{_name_prefix}_prompt.txt")
        with open(_meta_path, 'w', encoding='utf-8') as _mf:
            _mf.write(f"Step: {step}\n")
            _mf.write(f"Rank: {distributed_rank}\n")
            _mf.write(f"Video ID: {video_id}\n")
            _mf.write(f"Prompt: {prompt}\n")
            _mf.write(f"CFG Scale: {cfg_scale}\n")
            _mf.write(f"Sampling Steps: {diffusion_sampling_steps}\n")
            _mf.write(f"Num Frames: {num_frames}\n")
            _mf.write(f"Resolution: {width}x{height}\n")
            _mf.write(f"FPS: {fps}\n")
            _mf.write(f"Source: {source}\n")
            _mf.write(f"Task Type: {task_type}\n")
            _ck = camera_kwargs or {}
            _has_disc = 'action_labels' in _ck
            _mf.write(f"Camera Injected: {_has_disc} (discrete_cam_text={_has_disc})\n")
            _mf.write(f"i2v Conditioning: {cond_latent is not None}\n")
            _mf.write(f"Conditioning Latent Frames: {cond_latent_frames}\n")
            # record the per-frame discrete camera text + caption actually fed to the model
            if _has_disc:
                from wan.modules.model import ACTION_TEXT_TABLE
                _al = _ck['action_labels']
                if hasattr(_al, 'tolist'):
                    _al = _al.tolist()
                while isinstance(_al, (list, tuple)) and len(_al) > 0 and isinstance(_al[0], (list, tuple)):
                    _al = _al[0]
                _mf.write(f"\n===== Camera control text (actually used) =====\n")
                _mf.write(f"Action labels (per latent frame, 0~80): {_al}\n")
                _mf.write(f"Per-segment camera text (prepended before the caption and fed into cross-attn):\n")
                _i, _n = 0, len(_al)
                while _i < _n:
                    _j = _i
                    while _j < _n and _al[_j] == _al[_i]:
                        _j += 1
                    _lab = int(_al[_i])
                    _txt = (ACTION_TEXT_TABLE[_lab] if (ACTION_TEXT_TABLE and 0 <= _lab < len(ACTION_TEXT_TABLE))
                            else "(ACTION_TEXT_TABLE unavailable)")
                    _rng = f"frame {_i}" if (_j - _i == 1) else f"frame {_i}-{_j - 1}"
                    _mf.write(f"  [{_rng}] label={_lab}: {_txt}\n")
                    _i = _j
                # Example of actual per-frame input (first segment): camera_text + caption
                _lab0 = int(_al[0])
                _cam0 = (ACTION_TEXT_TABLE[_lab0] if (ACTION_TEXT_TABLE and 0 <= _lab0 < len(ACTION_TEXT_TABLE)) else "")
                _mf.write(f"\nFull text fed to model (frame 0 example) = [camera_text] + [caption]:\n")
                _mf.write(f"  {_cam0} {prompt}\n")

        # Joystick overlay for camera-controlled videos
        _val_action_labels = None
        if camera_kwargs and 'action_labels' in camera_kwargs:
            _val_action_labels = camera_kwargs['action_labels']

        if gt_frames is not None:
            if _val_action_labels is not None:
                from pipelines.common.control_overlay import superimpose_control_video as add_control_overlay
                left = add_control_overlay(gt_frames, _val_action_labels)
                right = add_control_overlay(frames, _val_action_labels)
                combined_path = os.path.join(output_dir, f"combined_control_{_name_prefix}.mp4")
            else:
                left, right = gt_frames, frames
                combined_path = os.path.join(output_dir, f"combined_{_name_prefix}.mp4")
            stitch_videos_with_labels(left, right, combined_path, "GT", "Generated", fps=fps)


# =============================================================================
# Checkpoint Resume
# ===========================================================================
# Main function
# =============================================================================
def run_training(args: argparse.Namespace) -> None:
    global DEFAULT_VIDEO_AREA_CAP, EVAL_INTERVAL, DIFFUSION_STEP_COUNT, CFG_GUIDANCE_WEIGHT

    # CUDA optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # Global variables
    DEFAULT_VIDEO_AREA_CAP = args.num_height * args.num_width
    EVAL_INTERVAL = getattr(args, 'validation_interval', 50)
    DIFFUSION_STEP_COUNT = getattr(args, 'diffusion_sampling_steps', 50)
    CFG_GUIDANCE_WEIGHT = getattr(args, 'cfg_scale', 5.0)

    # === Distributed init ===
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        rank=global_rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=1800),
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    current_device = torch.cuda.current_device()

    # SP init
    sp_size = getattr(args, 'sp_size', 1)
    setup_sequence_parallel_state(sp_size)

    # CP init
    cp_size = getattr(args, 'cp_size', 1)
    if cp_size > 1:
        setup_context_parallel(cp_size)
        if global_rank == 0:
            print(f"[Context Parallel] cp_size={cp_size}")
        from pipelines.utils.context_parallel import prime_context_parallel
        prime_context_parallel(hidden_dim=3072, num_heads=24, device=current_device)

    echo_on_main_rank(f"{'='*60}")
    echo_on_main_rank(f"Wan 2.2 5B Training (Advanced Pipeline)")
    echo_on_main_rank(f"World size: {world_size}, Local rank: {local_rank}")
    echo_on_main_rank(f"{'='*60}")

    # === Node-shared text encoding groups ===
    num_gpus_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', torch.cuda.device_count()))
    num_nodes = world_size // num_gpus_per_node
    node_id = global_rank // num_gpus_per_node
    gloo_timeout = datetime.timedelta(seconds=1800)
    node_group = None
    for n in range(num_nodes):
        ranks_in_node = list(range(n * num_gpus_per_node, (n + 1) * num_gpus_per_node))
        group = dist.new_group(ranks_in_node, backend='gloo', timeout=gloo_timeout)
        if n == node_id:
            node_group = group
    node_leader_rank = node_id * num_gpus_per_node

    args.node_group = node_group
    args.local_rank = local_rank
    args.num_local_ranks = num_gpus_per_node
    args.node_leader_rank = node_leader_rank
    echo_on_main_rank(f"--> Node groups (gloo): {num_nodes} nodes x {num_gpus_per_node} GPU/node")

    # === Seed ===
    if args.seed is not None:
        set_seed(args.seed + global_rank)

    # === Output dir ===
    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=4, default=str)
    dist.barrier()

    # TensorBoard
    tb_writer = None
    if global_rank == 0:
        tb_log_dir = os.path.join(args.output_dir, "tensorboard_logs")
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        args_text = "\n".join(f"  {k}: {v}" for k, v in sorted(vars(args).items()))
        tb_writer.add_text("config/training_args", f"```\n{args_text}\n```", 0)

    # === Model init ===
    transformer, vae_encoder, vae_decoder, text_encoder, tokenizer, encode_fn, history_encoder =\
        setup_wan_model(args, local_rank, global_rank)

    # Text encoder offload + sharing
    offload_te = getattr(args, 'offload_text_encoder', False)
    if offload_te:
        if text_encoder is not None:
            if hasattr(text_encoder, 'model'):
                text_encoder.model.to('cpu')
            elif hasattr(text_encoder, 'to'):
                text_encoder.to('cpu')
            torch.cuda.empty_cache()
            echo_on_main_rank("--> Text encoder offloaded to CPU (saves ~11GB)")
    if local_rank != 0:
        del text_encoder
        text_encoder = None
        free_gpu_memory()
        print(f"[Rank {global_rank}] Released text_encoder (shared from local_rank 0)")

    # Note: torch.compile on Wan VAE is NOT supported
    # (causal conv cache conflicts with CUDAGraphs)

    # === Resume ===
    resumed_step = 0
    if args.resume_from_checkpoint:
        resumed_step = read_checkpoint_for_resume(
            args.resume_from_checkpoint, transformer, global_rank,
        )
        resume_skip_data = getattr(args, 'resume_skip_data', 'True').lower() == 'true'
        if not resume_skip_data:
            echo_on_main_rank(f"[Resume] Loaded weights from step {resumed_step}, reset to 0")
            resumed_step = 0
    dist.barrier()

    # === trainable_modules (partial fine-tuning) ===
    _TRAINABLE_MODULE_MAP = {
        'camera_control': ['action_embedder', 'additive_camera_adapter'],
        'patch_embedding': ['patch_embedding'],
        'self_attn': ['self_attn'],
        'cross_attn': ['cross_attn'],
        'feedforward': ['ffn'],
        'adaln': ['adaln'],
        'output_proj': ['head'],
    }
    trainable_modules_str = getattr(args, 'trainable_modules', None)
    if trainable_modules_str:
        requested_modules = [m.strip() for m in trainable_modules_str.split(',') if m.strip()]
        trainable_keywords = []
        for mod in requested_modules:
            if mod in _TRAINABLE_MODULE_MAP:
                trainable_keywords.extend(_TRAINABLE_MODULE_MAP[mod])
                echo_on_main_rank(f"[trainable_modules] '{mod}' -> {_TRAINABLE_MODULE_MAP[mod]}")
            else:
                trainable_keywords.append(mod)
                echo_on_main_rank(f"[trainable_modules] keyword: '{mod}'")
        for param in transformer.parameters():
            param.requires_grad = False
        unfrozen_count = 0
        for name, param in transformer.named_parameters():
            if any(kw in name for kw in trainable_keywords):
                param.requires_grad = True
                unfrozen_count += param.numel()
        total_params = sum(p.numel() for p in transformer.parameters())
        echo_on_main_rank(f"[trainable_modules] Total: {total_params:,}, Trainable: {unfrozen_count:,}")

    # Freeze DIT for camera-only training
    if getattr(args, 'freeze_dit_weights', False):
        camera_keywords = ['control_adapter']
        for name, param in transformer.named_parameters():
            is_camera = any(kw in name for kw in camera_keywords)
            param.requires_grad = is_camera
        frozen_count = sum(1 for p in transformer.parameters() if not p.requires_grad)
        trainable_count = sum(1 for p in transformer.parameters() if p.requires_grad)
        echo_on_main_rank(f"--> Frozen: {frozen_count}, Trainable: {trainable_count}")

    # === FSDP ===
    transformer.train()
    transformer = enclose_wan_model_with_fsdp(transformer, args, current_device)

    # === Optimizer ===
    optimizer, lr_scheduler = build_optimizer_and_scheduler(transformer, args, extra_params=None)

    # === Pre-training validation (Step -1) ===
    if not getattr(args, 'skip_validation_before_train', False):
        dist.barrier()
        echo_on_main_rank("=" * 60)
        echo_on_main_rank("Pre-training validation (Step -1)")
        echo_on_main_rank("=" * 60)
        validation_prompt = "A stylish woman strolls down a bustling Tokyo street."
        perform_validation_sampling(
            transformer=transformer,
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            encode_fn=encode_fn,
            prompt=validation_prompt,
            num_frames=args.num_frames,
            height=getattr(args, 'validation_height', args.num_height),
            width=getattr(args, 'validation_width', args.num_width),
            device=current_device,
            output_dir=os.path.join(args.output_dir, "validation"),
            step=-1,
            distributed_rank=global_rank,
            cfg_scale=CFG_GUIDANCE_WEIGHT,
            is_fsdp_model=True,
            diffusion_sampling_steps=DIFFUSION_STEP_COUNT,
            fps=args.fps,
        )
        echo_on_main_rank("Pre-training validation done")

    # === Data loading ===
    resume_skip_data = str(getattr(args, 'resume_skip_data', 'True')).lower() in ('true', '1', 'yes')
    skip_epochs = 0
    if resumed_step > 0 and resume_skip_data:
        skip_batches = int(resumed_step * args.gradient_accumulation_steps)
        estimated_batches_per_epoch = max(1, 100000 // (args.train_batch_size * world_size))
        skip_epochs = skip_batches // estimated_batches_per_epoch

    train_dataloader, train_sampler, num_training_samples = build_train_loader(args, skip_epochs=skip_epochs)

    steps_per_epoch = max(1, num_training_samples // (args.train_batch_size * world_size * args.gradient_accumulation_steps))
    total_epochs = args.max_train_epochs
    total_train_steps = total_epochs * steps_per_epoch

    echo_on_main_rank(f"Dataset samples: {num_training_samples}")
    echo_on_main_rank(f"Steps per epoch: {steps_per_epoch}")
    echo_on_main_rank(f"Total train steps: {total_train_steps}")

    # === Training loop ===
    global_step = resumed_step
    start_epoch = resumed_step // steps_per_epoch if resumed_step > 0 else 0
    start_batch_in_epoch = resumed_step % steps_per_epoch if resumed_step > 0 else 0

    if resumed_step > 0 and resume_skip_data:
        train_sampler.set_skip_batches(start_batch_in_epoch, batch_size=1)
        echo_on_main_rank(f"[Resume] From epoch {start_epoch}, batch {start_batch_in_epoch}")

    progress_bar = tqdm(
        range(resumed_step, total_train_steps),
        desc="Wan2.2 Training",
        disable=local_rank > 0,
        initial=resumed_step,
        total=total_train_steps,
    )

    transformer.train()
    sigma_history = []
    SIGMA_HIST_INTERVAL = 50
    last_camera_kwargs = {}

    dist.barrier()

    for epoch in range(start_epoch, total_epochs):
        train_sampler.set_epoch(epoch)
        echo_on_main_rank(f"\n[Epoch {epoch}/{total_epochs}] (global_step={global_step})")

        _dataloader_iter = iter(train_dataloader)
        for batch_idx in range(len(train_dataloader)):
            data_item = next(_dataloader_iter, _DATALOADER_END)
            if data_item is _DATALOADER_END:
                break

            if global_step >= total_train_steps:
                break

            step_start_time = time.time()

            # Set current step on every module that tracks it (camera warmup, packforcing Mem ratio)
            _relative_step = global_step - resumed_step
            for _blk in transformer.modules():
                if hasattr(_blk, '_current_train_step'):
                    _blk._current_train_step = _relative_step
            # after FSDP wrap the real WanModel is at _fsdp_wrapped_module
            _root = transformer
            if hasattr(_root, '_fsdp_wrapped_module'):
                _root = _root._fsdp_wrapped_module
            if hasattr(_root, 'module'):
                _root = _root.module
            _root._current_train_step = global_step

            (step_loss, gradient_norm, last_prompt, last_camera_kwargs,
             last_task_type, last_cond_end, last_cond_latent, last_gt_video_latent,
             last_source, last_caption_type, last_K_ctrl, last_c2w_ctrl,
             last_K_ctrl_raw, last_c2w_ctrl_raw, last_video_id,
             last_sigma, last_gt_pixel_frames) = perform_single_training_step(
                transformer=transformer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                vae_encoder=vae_encoder,
                text_encoder=text_encoder,
                encode_fn=encode_fn,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                device=current_device,
                args=args,
                current_step=global_step,
                distributed_rank=global_rank,
                world_size=world_size,
                max_gradient_norm=args.max_grad_norm,
                data_item_override=data_item,
                dataloader_iter=_dataloader_iter,   # ★ real gradient accumulation: each micro-step pulls a different batch
                vae_decoder=vae_decoder,
            )

            step_duration = time.time() - step_start_time

            # Skip zero-loss batches (unless validation_only)
            validation_only_mode = getattr(args, 'validation_only', False)
            if step_loss == 0.0 and gradient_norm == 0.0 and not validation_only_mode:
                continue

            # Progress bar
            progress_bar.set_postfix({
                "loss": f"{step_loss:.4f}",
                "epoch": f"{epoch}/{total_epochs}",
                "task": last_task_type,
                "time": f"{step_duration:.2f}s",
                "grad": f"{gradient_norm:.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
            })
            progress_bar.update(1)

            # TensorBoard
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", step_loss, global_step)
                tb_writer.add_scalar("train/grad_norm", gradient_norm, global_step)
                tb_writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalar("train/step_time", step_duration, global_step)
                tb_writer.add_scalar("train/epoch", epoch, global_step)
                tb_writer.add_scalar(f"train/loss_{last_task_type}", step_loss, global_step)
                if last_cond_end > 0:
                    tb_writer.add_scalar("train/cond_frames", last_cond_end, global_step)
                _adv = getattr(transformer, '_adv_trainer', None)
                if _adv is not None and hasattr(_adv, '_last_d_loss'):
                    tb_writer.add_scalar("train/d_loss", _adv._last_d_loss, global_step)
                if _adv is not None and hasattr(_adv, '_last_g_loss'):
                    tb_writer.add_scalar("train/g_loss", _adv._last_g_loss, global_step)
                tb_writer.add_scalar("train/sigma", last_sigma, global_step)
                sigma_history.append(last_sigma)
                if len(sigma_history) >= SIGMA_HIST_INTERVAL:
                    tb_writer.add_histogram("train/sigma_distribution", torch.tensor(sigma_history), global_step)
                    sigma_history.clear()
                if last_source:
                    src = str(last_source) if not isinstance(last_source, str) else last_source
                    tb_writer.add_scalar(f"train/loss_source/{src}", step_loss, global_step)

            # === Validation ===
            _first_val_step = getattr(args, 'first_validation_step', 0)
            _resume_buffer = _first_val_step if (_first_val_step > 0 and resumed_step > 0) else 0
            _should_validate = (global_step % EVAL_INTERVAL == 0 and global_step > 0
                                and global_step != resumed_step
                                and global_step >= resumed_step + _resume_buffer)
            if _first_val_step > 0:
                if resumed_step == 0 and global_step == _first_val_step:
                    _should_validate = True   # first val on cold start
                elif resumed_step > 0 and global_step == resumed_step + _first_val_step:
                    _should_validate = True   # first val after the resume buffer
            if validation_only_mode or _should_validate:
                echo_on_main_rank(f"\n{'='*60}")
                echo_on_main_rank(f"[Validation] Step {global_step}")
                echo_on_main_rank(f"{'='*60}")

                free_gpu_memory()

                validation_prompt = last_prompt if last_prompt else "A beautiful sunset over the ocean."
                val_height = getattr(args, 'validation_height', args.num_height)
                val_width = getattr(args, 'validation_width', args.num_width)

                # Validation frames: use GT video's actual frame count so camera/GT latent stay aligned
                if last_gt_video_latent is not None:
                    gt_latent_T = last_gt_video_latent.shape[1]
                    val_frames = 1 + (gt_latent_T - 1) * WAN_VAE_TIME_STRIDE
                else:
                    val_frames = getattr(args, 'validation_frames', args.num_frames)
                vf_tensor = torch.tensor([val_frames], device=current_device)
                dist.all_reduce(vf_tensor, op=dist.ReduceOp.MAX)
                val_frames = int(vf_tensor.item())

                val_cond_latent = None
                val_cond_end = 0
                use_i2v = (last_cond_latent is not None and
                           (last_task_type in ("i2v", "v2v") or args.training_mode in ("i2v", "hybrid")))
                if use_i2v:
                    val_cond_end = last_cond_end
                    val_cond_latent = last_cond_latent.to(device=current_device, dtype=torch.bfloat16)

                val_camera_kwargs = None
                if getattr(args, 'use_camera_control', False) and last_camera_kwargs:
                    val_camera_kwargs = {}
                    if 'action_labels' in last_camera_kwargs:
                        val_camera_kwargs['action_labels'] = last_camera_kwargs['action_labels'].to(
                            device=current_device).contiguous()

                val_gt = last_gt_video_latent
                if last_gt_video_latent is not None:
                    val_gt = last_gt_video_latent.unsqueeze(0) if last_gt_video_latent.dim() == 4 else last_gt_video_latent

                if offload_te and text_encoder is not None:
                    _te_mod = getattr(text_encoder, 'model', text_encoder)
                    _te_mod.to(current_device)

                perform_validation_sampling(
                    transformer=transformer,
                    vae_decoder=vae_decoder,
                    text_encoder=text_encoder,
                    encode_fn=encode_fn,
                    prompt=validation_prompt,
                    num_frames=val_frames,
                    height=val_height,
                    width=val_width,
                    device=current_device,
                    output_dir=os.path.join(args.output_dir, "validation"),
                    step=global_step,
                    distributed_rank=global_rank,
                    cfg_scale=CFG_GUIDANCE_WEIGHT,
                    is_fsdp_model=True,
                    camera_kwargs=val_camera_kwargs,
                    diffusion_sampling_steps=DIFFUSION_STEP_COUNT,
                    task_type=last_task_type,
                    cond_latent=val_cond_latent,
                    cond_latent_frames=val_cond_end,
                    gt_video_latent=val_gt,
                    video_id=last_video_id,
                    fps=args.fps,
                    source=str(last_source) if last_source else "unknown",
                    gt_pixel_frames=last_gt_pixel_frames,
                )

                if offload_te and text_encoder is not None:
                    _te_mod = getattr(text_encoder, 'model', text_encoder)
                    _te_mod.to('cpu')
                    free_gpu_memory()

                echo_on_main_rank(f"[Validation] Step {global_step}: done\n")

                if validation_only_mode:
                    echo_on_main_rank(f"\n[validation_only] Done, exiting.")
                    dist.barrier()
                    dist.destroy_process_group()
                    sys.exit(0)

            # === Checkpoint ===
            should_save_ckpt = (global_step % args.checkpointing_steps == 0 and global_step > 0)
            if should_save_ckpt:
                echo_on_main_rank(f"[Checkpoint] Saving step={global_step}")
                free_gpu_memory()
                store_checkpoint(transformer, global_rank, args.output_dir, global_step)
                dist.barrier()
                echo_on_main_rank(f"[Checkpoint] Saved step={global_step}")

            free_gpu_memory()
            global_step += 1

        if global_step >= total_train_steps:
            echo_on_main_rank(f"[Epoch {epoch}] Reached total steps {total_train_steps}")
            break

    # Final checkpoint
    store_checkpoint(transformer, global_rank, args.output_dir, global_step)

    # Cleanup
    if tb_writer is not None:
        tb_writer.close()

    if fetch_sequence_parallel_state():
        teardown_sequence_parallel_group()
    if cp_is_active():
        teardown_context_parallel()

    if dist.is_initialized():
        dist.barrier()
    echo_on_main_rank("--> Wan 2.2 training complete!")


# =============================================================================
# Command-line arguments
# =============================================================================
def parse_cli_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan 2.2 5B Training (Advanced Pipeline)")

    # ===== Model paths =====
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--text_encoder_path", type=str, default=None)
    parser.add_argument("--wan_model_type", type=str, default="ti2v", choices=["t2v", "i2v", "ti2v"])
    parser.add_argument("--wan_version", type=str, default="auto", choices=["auto", "2.1", "2.2"],
                        help="Wan backbone: 2.1 (1.3B) or 2.2 (5B). auto = detect from --pretrained_model_path")

    # ===== Dataset =====
    parser.add_argument("--num_height", type=int, default=544)
    parser.add_argument("--num_width", type=int, default=960)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2,
                       help="number of batches each worker prefetches, total prefetch = num_workers × prefetch_factor")

    # ===== BiWM discrete camera-text dataset (dataset/videos, pose parsed from directory name/json into latent-level discrete actions) =====
    parser.add_argument("--biwm_video_dir", type=str, default=None,
                        help="dataset/videos directory (each subdirectory contains gen.mp4)")
    parser.add_argument("--biwm_caption_json", type=str, default=None,
                        help="videos_syn.json (list, index==6-digit id, contains caption + action_frames)")

    # ===== Training config =====
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # ===== LR Scheduler =====
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=20)

    # ===== Resume =====
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--resume_skip_data", type=str, default="True")

    # ===== Training mode =====
    parser.add_argument("--training_mode", type=str, default="t2v", choices=["t2v", "i2v", "v2v", "hybrid"])
    parser.add_argument("--i2v_cond_latent_frames", type=int, default=1)
    parser.add_argument("--v2v_cond_latent_frames", type=int, default=0)
    parser.add_argument("--v2v_cond_ratio", type=float, default=0.25)
    parser.add_argument("--v2v_cond_ratio_min", type=float, default=None,
                       help="Min v2v conditioning ratio (random sample). If set with max, overrides v2v_cond_ratio")
    parser.add_argument("--v2v_cond_ratio_max", type=float, default=None,
                       help="Max v2v conditioning ratio (random sample)")
    parser.add_argument("--i2v_prob", type=float, default=0.3)
    parser.add_argument("--v2v_prob", type=float, default=0.2)

    # ===== Progressive frames =====

    # ===== Multi-resolution =====
    parser.add_argument("--high_res_height", type=int, default=720)
    parser.add_argument("--high_res_width", type=int, default=1280)
    parser.add_argument("--high_res_prob", type=float, default=0.0)
    parser.add_argument("--high_res_step", type=int, default=0)

    # ===== Camera control =====
    parser.add_argument("--use_camera_control", action="store_true")
    parser.add_argument("--num_actions", type=int, default=81)
    parser.add_argument("--no_discrete_camera", action="store_true")
    parser.add_argument("--train_sigma_shift", type=float, default=0.0,
                        help="Shift applied to the TRAINING sigma: shifted = s*σ/(1+(s-1)*σ). 0=uniform, >0=bias toward high-noise.")
    parser.add_argument("--val_sigma_shift", type=float, default=5.0,
                        help="Shift applied to the VALIDATION/sampling sigma schedule (Wan default ~3-5).")
    parser.add_argument("--freeze_dit_weights", action="store_true")

    # ===== trainable_modules =====
    parser.add_argument("--trainable_modules", type=str, default=None)

    # ===== FSDP =====
    parser.add_argument("--use_cpu_offload", action="store_true")

    # ===== Validation =====
    parser.add_argument("--validation_only", action="store_true")
    parser.add_argument("--validation_interval", type=int, default=50)
    parser.add_argument("--first_validation_step", type=int, default=0)
    parser.add_argument("--skip_validation_before_train", action="store_true", default=False)
    parser.add_argument("--diffusion_sampling_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--validation_height", type=int, default=544)
    parser.add_argument("--validation_width", type=int, default=960)
    parser.add_argument("--validation_frames", type=int, default=81)

    # ===== Output =====
    parser.add_argument("--output_dir", type=str, default="./outputs_wan_22")
    parser.add_argument("--checkpointing_steps", type=int, default=500)

    # ===== Parallel =====
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--cp_size", type=int, default=1)

    # ===== Memory optimization =====
    parser.add_argument("--offload_text_encoder", action="store_true")
    parser.add_argument("--torch_compile", action="store_true", help="torch.compile VAE for memory savings")

    # ===== Debug =====

    register_lora_args(parser)

    args = parser.parse_args()
    return args


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    args = parse_cli_options()
    run_training(args)
