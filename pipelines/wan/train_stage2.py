#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BiWM Stage 2 — DMD distillation (Wan2.2 5B training entry, --dmd_distill).

Supports cam-text conditioned video generation, FSDP distributed training,
bf16 mixed precision, Context Parallel, multi-resolution, and partial fine-tuning.
"""
# isort: skip_file
import argparse
import datetime
import gc
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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

# Wan model imports (use wan/ directory, independent from ltx2)
from wan.modules.model import WanModel, WanAttentionBlock

# Local module imports
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
    fetch_cp_rank,
    disperse_sequence,
    collect_sequence,
    collect_for_loss,
    calc_cp_divisible_frames,
    pad_for_cp_divisible,
    strip_cp_padding,
)

# Camera control module


# =============================================================================
# Constant definitions
# =============================================================================
# Wan 2.2 VAE parameters

# Wan2.2 official negative prompt (used for CFG)

# Wan2.2-TI2V-5B actual model config

# Wan2.2 VAE normalization parameters (48-channel)

# Training-related global variables


# =============================================================================
# CheckpointableDistSampler
# ===========================================================================
# Utility functions
# =============================================================================
# Wan2.1-1.3B VAE parameters (16-channel, 8x spatial) — alternative backbone


# =============================================================================
# Video saving
# ===========================================================================
# VAE encode/decode (Wan 2.2)
# ===========================================================================
# Model loading
# ===========================================================================
# FSDP wrapping
# ===========================================================================
# Optimizer and scheduler
# ===========================================================================
# Data loading
# ===========================================================================
# Intra-node shared text encoding (all_gather + leader does the encoding + each rank retrieves its own result)
# ===========================================================================
# Training step
# =============================================================================
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

        data_item = data_item_override

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

            # Dataset already normalizes to [-1, 1]; clamp after bicubic resize overshoot.
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
        # Optional: apply shift to sigma (bias toward high-noise region, aligned with inference schedule)
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

        # === BiWM discrete action_labels (precomputed from action_frames) ===
        if _use_cam and precomp_action_labels is not None:
            _al = precomp_action_labels
            if not isinstance(_al, torch.Tensor):
                _al = torch.as_tensor(_al)
            if _al.dim() == 1:
                _al = _al.unsqueeze(0)
            _al = _al.to(device=device, dtype=torch.long)
            if _al.shape[1] != T_lat:
                _al = F.interpolate(_al.float().unsqueeze(1), size=T_lat, mode='nearest').squeeze(1).long()
            camera_kwargs['action_labels'] = _al

        # CP sync camera_kwargs
        if cp_is_active():
            _cam_flag = torch.tensor([1.0 if camera_kwargs else 0.0], device=device)
            dist.broadcast(_cam_flag, src=cp_leader, group=cp_cfg.cp_group)
            if _cam_flag.item() <= 0.5:
                camera_kwargs = {}

        # Barrier before forward
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

        # === Adversarial Loss (optional) ===
        adv_trainer = getattr(transformer, '_adv_trainer', None)
        if adv_trainer is not None:
            sigma_val = sigma.item()
            if cond_end > 0:
                noisy_slice = noisy_latent[:, cond_end:].unsqueeze(0)
                fake_x0 = noisy_slice - sigma_val * output_for_loss
                real_x0 = video_latent[:, cond_end:].unsqueeze(0)
            else:
                noisy_full = noisy_latent.unsqueeze(0)
                fake_x0 = noisy_full - sigma_val * video_output
                real_x0 = video_latent.unsqueeze(0)
            d_loss = adv_trainer.discriminator_step(real_x0.detach(), fake_x0.detach())
            g_loss = adv_trainer.generator_loss(fake_x0)
            loss = loss + adv_trainer.gan_weight * g_loss
            adv_trainer._last_d_loss = d_loss
            adv_trainer._last_g_loss = g_loss.item()
            log_main(f"[GAN] d={d_loss:.4f}, g={g_loss.item():.4f}")

        # === NaN check (global sync) ===
        local_nan = torch.tensor(
            [1.0 if (torch.isnan(loss) or torch.isinf(loss)) else 0.0],
            device=device,
        )
        dist.all_reduce(local_nan, op=dist.ReduceOp.MAX)
        if local_nan.item() > 0.5:
            log_main(f"[WARNING] NaN/Inf loss detected, skip step")
            optimizer.zero_grad()
            continue

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
            try:
                with open(_anomaly_log, 'a', encoding='utf-8') as _af:
                    _af.write(f"step={current_step}, loss={_loss_val:.4f}, "
                              f"video_id={video_id}, source={source}\n")
            except Exception:
                pass
            optimizer.zero_grad()
            continue

        # Backward (catch cuDNN/CUDA transient errors)
        try:
            loss.backward()
        except RuntimeError as _bwd_err:
            _bwd_msg = str(_bwd_err)
            log_main(f"[WARNING] backward failed: {_bwd_msg[:200]}")
            # Sync skip across all ranks to avoid NCCL hang
            _skip_flag = torch.tensor([1.0], device=device)
            dist.all_reduce(_skip_flag, op=dist.ReduceOp.MAX)
            optimizer.zero_grad()
            del video_output, loss, noisy_latent, noise, video_latent, context
            free_gpu_memory()
            continue

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

        accumulated_loss += loss.detach().item()

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
    stg_scale: float = 0.0,
    stg_blocks: Optional[List[int]] = None,
    rescale_scale: float = 0.7,
    source: str = "unknown",
    gt_pixel_frames=None,
) -> None:
    dist.barrier()
    was_training = transformer.training

    try:
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
                    stg_scale=stg_scale,
                    stg_blocks=stg_blocks,
                    gt_pixel_frames=gt_pixel_frames,
                    rescale_scale=rescale_scale,
                    source=source,
                )

        _cp_settings.enabled = cp_was_enabled
    except Exception as e:
        print(f"[Rank {distributed_rank}] Validation error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
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
    stg_scale=0.0,
    stg_blocks=None,
    rescale_scale=0.7,
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

    # Decode and save
    if on_main_rank or True:  # All ranks decode for FSDP
        try:
            os.makedirs(output_dir, exist_ok=True)
            _safe_vid = str(video_id).replace("/", "_").replace("\\", "_").replace(" ", "_")[:80]
            _step_str = "pretrain" if step < 0 else f"{step:06d}"
            _safe_source = str(source).replace("/", "_").replace(" ", "_")[:30]
            _has_cam_tag = "cam" if (camera_kwargs and ('action_labels' in camera_kwargs
                                     or 'action_labels' in camera_kwargs)) else "nocam"
            _name_prefix = f"step_{_step_str}_{task_type}_{_has_cam_tag}_{_safe_source}_rank_{distributed_rank:03d}_{_safe_vid}"

            # Decode generated video
            frames = wan_vae_reconstruct(vae_decoder, video_latent.squeeze(0), device)

            # GT: decode gt_video_latent through VAE to see VAE reconstruction quality
            gt_frames = None
            if gt_video_latent is not None:
                try:
                    _gt_lat = gt_video_latent.squeeze(0) if gt_video_latent.dim() == 5 else gt_video_latent
                    gt_frames = wan_vae_reconstruct(vae_decoder, _gt_lat, device)
                except Exception as e:
                    print(f"[Rank {distributed_rank}] GT VAE decode failed: {e}, falling back to pixel frames")
                    gt_frames = gt_pixel_frames
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
            try:
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
                    _mf.write(f"Camera Injected: {bool(camera_kwargs and 'action_labels' in camera_kwargs)} "
                              f"(discrete_cam_text={'action_labels' in (camera_kwargs or {})})\n")
                    _mf.write(f"i2v Conditioning: {cond_latent is not None}\n")
                    _mf.write(f"Conditioning Latent Frames: {cond_latent_frames}\n")
            except Exception:
                pass

            # Joystick overlay for camera-controlled videos
            _val_action_labels = None
            if camera_kwargs and 'action_labels' in camera_kwargs:
                _val_action_labels = camera_kwargs['action_labels']

            if gt_frames is not None:
                try:
                    if _val_action_labels is not None:
                        from pipelines.common.control_overlay import superimpose_control_video as add_control_overlay
                        left = add_control_overlay(gt_frames, _val_action_labels)
                        right = add_control_overlay(frames, _val_action_labels)
                        combined_path = os.path.join(output_dir, f"combined_control_{_name_prefix}.mp4")
                    else:
                        left, right = gt_frames, frames
                        combined_path = os.path.join(output_dir, f"combined_{_name_prefix}.mp4")
                    stitch_videos_with_labels(left, right, combined_path, "GT", "Generated", fps=fps)
                except Exception as e:
                    print(f"[Rank {distributed_rank}] Combined video failed: {e}")
                    # Fallback without control
                    combined_path = os.path.join(output_dir, f"combined_{_name_prefix}.mp4")
                    stitch_videos_with_labels(gt_frames, frames, combined_path, "GT", "Generated", fps=fps)
        except Exception as e:
            print(f"[Rank {distributed_rank}] Validation save failed: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# Checkpoint Resume
# ===========================================================================
# HR Pretrain checkpoint helpers (save/inherit history_encoder)
# =============================================================================
def _store_history_encoder(history_encoder, rank, output_dir, step):
    """Save history_encoder.safetensors to checkpoint-{step}/ (for Stage 2 to inherit).
    history_encoder is not FSDP-wrapped, so rank 0 can directly get its state_dict."""
    if rank > 0 or history_encoder is None:
        return
    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in history_encoder.state_dict().items()}
    he_path = os.path.join(save_dir, "history_encoder.safetensors")
    safetensors.torch.save_file(sd, he_path)
    print(f"--> history_encoder saved: {he_path} ({len(sd)} tensors)", flush=True)


def _fold_lora_into_base(state_dict, scaling=1.0):
    """Merge LoRA ckpt → full-parameter base state_dict.
    Input: state_dict containing the {prefix.base.weight, prefix.base.bias, prefix.lora_A, prefix.lora_B} quadruple.
    Output: {prefix.weight = base.weight + scaling * (lora_B @ lora_A), prefix.bias = base.bias}; lora tensors discarded.
    Non-LoRA-wrapped keys (e.g. patch_embedding.weight) are passed through directly.
    used to load a LoRA-ckpt into a non-LoRA (full-parameter) transformer."""
    lora_prefixes = sorted({k[:-len('.lora_A')] for k in state_dict if k.endswith('.lora_A')})
    if not lora_prefixes:
        return state_dict
    merged = {}
    handled = set()
    for prefix in lora_prefixes:
        _kw, _ka, _kb = f"{prefix}.base.weight", f"{prefix}.lora_A", f"{prefix}.lora_B"
        base_w = state_dict[_kw]; lora_a = state_dict[_ka]; lora_b = state_dict[_kb]
        assert base_w.shape == (lora_b.shape[0], lora_a.shape[1]),\
            f"LoRA shape mismatch @ {prefix}: base={base_w.shape}, B={lora_b.shape}, A={lora_a.shape}"
        delta = scaling * (lora_b.float() @ lora_a.float())
        merged[f"{prefix}.weight"] = (base_w.float() + delta).to(base_w.dtype)
        bias_k = f"{prefix}.base.bias"
        if bias_k in state_dict:
            merged[f"{prefix}.bias"] = state_dict[bias_k]
            handled.add(bias_k)
        handled.update({_kw, _ka, _kb})
    for k, v in state_dict.items():
        if k in handled:
            continue
        merged[k] = v
    return merged


def _carry_stage1_into_stage2(transformer, history_encoder, stage1_ckpt, device):
    """Stage 2/3 inherit ckpt: load transformer state + history_encoder.
    automatically detect the LoRA state of the ckpt and the target transformer:
      - target is still LowRankLinear (has .lora_A key) → load directly via original path (ckpt should also be LoRA-wrapped).
      - target is full-parameter plain Wan (no .lora_A key) while ckpt is LoRA-wrapped → merge LoRA into base then load.
    Must be called before FSDP wrap.
    """
    import os as _os
    _tf_path = _os.path.join(stage1_ckpt, "diffusion_pytorch_model.safetensors")
    _he_path = _os.path.join(stage1_ckpt, "history_encoder.safetensors")
    if not _os.path.exists(_tf_path):
        raise FileNotFoundError(f"Stage1 transformer ckpt does not exist: {_tf_path}")
    _tf_sd = safetensors.torch.load_file(_tf_path, device='cpu')
    _tgt_has_lora = any(k.endswith('.lora_A') for k in transformer.state_dict().keys())
    _ckpt_has_lora = any(k.endswith('.lora_A') for k in _tf_sd.keys())
    if _ckpt_has_lora and not _tgt_has_lora:
        _orig_n = len(_tf_sd)
        _tf_sd = _fold_lora_into_base(_tf_sd, scaling=1.0)
        print(f"[Stage2-Load] LoRA→base merge (alpha/rank=1.0): {_orig_n} → {len(_tf_sd)} tensors", flush=True)
    _missing, _unexpected = transformer.load_state_dict(_tf_sd, strict=False)
    print(f"[Stage2-Load] transformer <- {_tf_path}: "
          f"loaded {len(_tf_sd)} tensors, missing={len(_missing)}, unexpected={len(_unexpected)}", flush=True)
    if len(_unexpected) > 0:
        print(f"[Stage2-Load] ⚠ unexpected keys first 5: {_unexpected[:5]}", flush=True)
    # history_encoder
    # HE file is optional —— stage1 (train_stage1.py, non-autoregressive base) has no history_encoder,
    #   skip when history_encoder.safetensors does not exist (HE keeps random init, trained by DMD multi-block rollout),
    #   instead of raising. (Old logic required HE to exist, which only applies when inheriting from packforcing rollout_pretrain.)
    if history_encoder is not None:
        if not _os.path.exists(_he_path):
            print(f"[Stage2-Load] ⚠ history_encoder ckpt does not exist ({_he_path}); "
                  f"HE keeps random init, trained by DMD multi-block autoregression (no HE in stage1 base is normal).", flush=True)
        else:
            _he_sd = safetensors.torch.load_file(_he_path, device='cpu')
            _hm, _hu = history_encoder.load_state_dict(_he_sd, strict=False)
            print(f"[Stage2-Load] history_encoder <- {_he_path}: "
                  f"loaded {len(_he_sd)} tensors, missing={len(_hm)}, unexpected={len(_hu)}", flush=True)


# =============================================================================
# Stage 3: DMD distillation main loop (50-step → 4-step, bidirectional autoregression)
# =============================================================================
def _dmd_extract_cam(di):
    """Unpack caption + camera fields from data_item.
    - BiwmCamCaptionData: 16-field, last is precomputed discrete action_labels (latent-level, given directly to full_labels)
    """
    cap = di.get("caption", "")
    if isinstance(cap, (list, tuple)):
        cap = cap[0]
    if not isinstance(cap, str):
        cap = str(cap)
    has = di.get("has_camera", True)
    hc = (bool(has.any()) if isinstance(has, torch.Tensor) else bool(has)) if has is not None else False
    action_labels = di.get("action_labels")
    # continuous camera (K_ctrl/c2w/c2w_raw) removed; pose_orig kept 0.0 for return-signature compat
    return cap, None, None, None, hc, 0.0, 0.0, action_labels


def _dmd_embed_gt_latent(video, vae_encoder, device, n_latent_frames, expect_chw=None):
    """di[0] real video → VAE latent [1,48,n_latent_frames,Hlat,Wlat] (GAN real / GT regression).
    video: [T,C,H,W] or [1,T,C,H,W], pixels in [-1,1]. If frame count < n_latent_frames or channel/spatial shape does not
    match x0_gen (dataset resolution bucketing) → return None (skip this step, prevent crash from shape mismatch)."""
    try:
        v = video[0] if isinstance(video, (list, tuple)) else video
        if torch.is_tensor(v) and v.dim() == 5:
            v = v[0]
        if not torch.is_tensor(v) or v.dim() != 4:
            return None
        v = v.permute(1, 0, 2, 3).contiguous().to(device).float().clamp(-1.0, 1.0)   # [C,T,H,W]
        T = v.shape[1]
        if (T - 1) % 4 != 0:                       # Wan VAE 4x temporal downsampling, T must satisfy (T-1)%4==0
            T = 1 + 4 * ((T - 1) // 4)
            v = v[:, :T]
        with torch.no_grad():
            lat = vae_encoder.encode([v])[0]       # [48, Tlat, Hlat, Wlat]
        lat = lat.unsqueeze(0)                      # [1,48,Tlat,Hlat,Wlat]
        if lat.shape[2] < n_latent_frames:
            return None
        if expect_chw is not None and tuple(lat.shape[1:]) != (int(expect_chw[0]), lat.shape[2],
                                                               int(expect_chw[1]), int(expect_chw[2])):
            # Channel/spatial resolution does not match x0_gen (bucketing/scaling), skip to avoid MSE/discriminator shape mismatch
            return None
        return lat[:, :, :n_latent_frames].contiguous()
    except Exception as _e:
        print(f"[DMD-GT] encode failed, skip GT for this step: {_e}", flush=True)
        return None


@torch.no_grad()
def _dmd_dump_text_log(output_dir, step, rank, caption_text, full_labels):
    """Save the text actually used by DMD validation (per-frame [discrete camera text] + caption) into a txt.
    In the model forward, each latent frame prepends ACTION_TEXT_TABLE[label] to the caption for cross-attn."""
    import os
    if rank != 0:
        return
    try:
        from wan.modules.model import ACTION_TEXT_TABLE
    except Exception:
        ACTION_TEXT_TABLE = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        _al = full_labels
        if hasattr(_al, 'tolist'):
            _al = _al.tolist()
        while isinstance(_al, (list, tuple)) and len(_al) > 0 and isinstance(_al[0], (list, tuple)):
            _al = _al[0]
        path = os.path.join(output_dir, f"dmd_val_step{step:08d}_rank{rank}_text.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"Step: {step}\n")
            f.write(f"Caption: {caption_text}\n")
            if not _al:
                f.write("Camera: (no discrete camera, pure t2v)\n")
                return
            f.write(f"\n===== Camera control text (actually used) =====\n")
            f.write(f"Action labels (per latent frame, 0~80): {_al}\n")
            f.write(f"Per-segment camera text (prepended before caption and fed to cross-attn):\n")
            i, n = 0, len(_al)
            while i < n:
                j = i
                while j < n and _al[j] == _al[i]:
                    j += 1
                lab = int(_al[i])
                txt = (ACTION_TEXT_TABLE[lab] if (ACTION_TEXT_TABLE and 0 <= lab < len(ACTION_TEXT_TABLE))
                       else "(ACTION_TEXT_TABLE unavailable)")
                rng = f"frame {i}" if (j - i == 1) else f"frame {i}-{j - 1}"
                f.write(f"  [{rng}] label={lab}: {txt}\n")
                i = j
            lab0 = int(_al[0])
            cam0 = (ACTION_TEXT_TABLE[lab0] if (ACTION_TEXT_TABLE and 0 <= lab0 < len(ACTION_TEXT_TABLE)) else "")
            f.write(f"\nFull text fed to model (frame 0 example) = [camera_text] + [caption]:\n")
            f.write(f"  {cam0} {caption_text}\n")
    except Exception as _e:
        print(f"[DMD-Val] text log write failed: {_e}", flush=True)


def _dmd_evaluate(generator, history_encoder, real_score, vae_decoder, args, cap_emb, neg_emb,
                  full_labels, latent_shape, device, dtype, output_dir, step, rank, fps,
                  caption_text=None):
    """DMD training-time validation: under the same caption+camera, generate side by side
      (1) generator 4-step (distilled student, autoregressive rollout of the full segment)
      (2) real teacher 50-step CFG (t2v full segment) —— ★ check whether the teacher signal is correct (generator converges toward it)
    decode + camera joystick + side-by-side comparison. 1 sample per rank, no_grad throughout."""
    import os
    from pipelines.wan.dmd_core import student_rollout, data_teacher_sample
    from pipelines.common.control_overlay import superimpose_control_video as add_control_overlay
    # record the text actually used in this validation (per-frame [discrete camera text] + caption)
    _dmd_dump_text_log(output_dir, step, rank, caption_text, full_labels)
    generator.eval()
    if history_encoder is not None:
        history_encoder.eval()

    def _decode_overlay(x0):
        _model = vae_decoder.model if hasattr(vae_decoder, 'model') else vae_decoder
        if hasattr(_model, 'clear_cache'):
            _model.clear_cache()
        pix = vae_decoder.decode([x0.squeeze(0).to(torch.float32)])[0]   # [3,T_pix,H,W]
        pix = ((pix.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        fr = [pix[i] for i in range(pix.shape[0])]
        if full_labels is not None:
            try:
                fr = add_control_overlay(fr, full_labels)   # standard joystick (causal mapping + smoothing + energy ring)
            except Exception as _je:
                print(f"[DMD-Val rank={rank}] control overlay failed: {_je}", flush=True)
        return fr

    try:
        os.makedirs(output_dir, exist_ok=True)
        # 1. generator 4-step rollout of the full segment
        x0_gen, _, _, _ = student_rollout(generator, history_encoder, args, cap_emb,
                                            full_labels, latent_shape, device, dtype, train_mode=False)
        gen_frames = _decode_overlay(x0_gen)
        store_video(gen_frames, os.path.join(output_dir, f"dmd_val_step{step:08d}_rank{rank}_gen4step.mp4"), fps=fps)

        # quantization-effect check: same seed, run gen with quant-OFF(bf16) vs quant-ON, save side-by-side video.
        _quant_set = None
        if getattr(args, 'dmd_nvfp4', False):
            from pipelines.common.nvfp4 import toggle_nvfp4_quantization as _quant_set
            _qtag = "NVFP4"
        elif getattr(args, 'dmd_fp8', False):
            from pipelines.common.fp8quant import toggle_fp8_quantization as _quant_set
            _qtag = "FP8"
        if _quant_set is not None:
            _rng = torch.random.get_rng_state()
            _quant_set(generator, False); torch.manual_seed(12345 + step)
            x0_bf16, _, _, _ = student_rollout(generator, history_encoder, args, cap_emb,
                                                 full_labels, latent_shape, device, dtype, train_mode=False)
            _quant_set(generator, True); torch.manual_seed(12345 + step)
            x0_q, _, _, _ = student_rollout(generator, history_encoder, args, cap_emb,
                                              full_labels, latent_shape, device, dtype, train_mode=False)
            torch.random.set_rng_state(_rng)
            if rank == 0:
                stitch_videos_with_labels(_decode_overlay(x0_bf16), _decode_overlay(x0_q),
                                           os.path.join(output_dir, f"dmd_val_step{step:08d}_rank{rank}_quantgap.mp4"),
                                           "Gen-BF16", f"Gen-{_qtag}", fps=fps)

        # 1b. diagnosis: generator runs 50-step CFG on the first block (no history, single block of K latent).
        #   FSDP: all ranks must run the generator forward (collective), only rank0 does decode/save.
        _K1 = int(args.dmd_block_K)
        _cfg_g = float(getattr(args, 'real_guidance_scale', 5.0))
        _steps_g = int(getattr(args, 'diffusion_sampling_steps', 50))
        x0_gen50 = data_teacher_sample(
            generator, cap_emb, neg_emb, None, latent_shape, _K1, device, dtype, args,
            num_steps=_steps_g, cfg=_cfg_g)
        if rank == 0:
            gen50_frames = _decode_overlay(x0_gen50)
            store_video(gen50_frames, os.path.join(output_dir, f"dmd_val_step{step:08d}_rank{rank}_gen50step_block1.mp4"), fps=fps)
            # Direct side-by-side: [4-step first block | 50-step first block] (take the first (K-1)*4+1 frames of the full 4-step video = first block)
            _pix_b1 = (_K1 - 1) * 4 + 1
            stitch_videos_with_labels(
                gen_frames[:_pix_b1], gen50_frames,
                os.path.join(output_dir, f"dmd_val_step{step:08d}_rank{rank}_block1_4step_vs_50step.mp4"),
                "Gen 4-step blk1", "Gen 50-step CFG blk1", fps=fps)

        # 2. real teacher 50-step CFG (t2v full segment, check teacher signal correctness)
        total_K = int(args.dmd_num_blocks) * int(args.dmd_block_K)
        _cfg = float(getattr(args, 'real_guidance_scale', 5.0))
        _steps = int(getattr(args, 'diffusion_sampling_steps', 50))
        x0_teacher = data_teacher_sample(
            real_score, cap_emb, neg_emb, full_labels, latent_shape, total_K, device, dtype, args,
            num_steps=_steps, cfg=_cfg)
        teacher_frames = _decode_overlay(x0_teacher)
        store_video(teacher_frames, os.path.join(output_dir, f"dmd_val_step{step:08d}_rank{rank}_teacher50step.mp4"), fps=fps)

        # 3. Side-by-side comparison [generator 4-step | teacher 50-step]
        cmp = os.path.join(output_dir, f"dmd_val_step{step:08d}_rank{rank}_cmp.mp4")
        stitch_videos_with_labels(gen_frames, teacher_frames, cmp, "Gen 4-step", "Teacher 50-step", fps=fps)
    except Exception as _e:
        import traceback
        print(f"[DMD-Val rank={rank}] failed: {_e}", flush=True)
        traceback.print_exc()
    generator.train()
    if history_encoder is not None:
        history_encoder.train()


def _dmd_gradient_norm(params):
    """Gradient L2 norm (before clip, FSDP local shard; used to see whether each part is learning / whether it explodes/is 0)."""
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().float().norm() ** 2)
    return sq ** 0.5


def execute_dmd_distillation(args, generator, history_encoder, vae_encoder, vae_decoder,
                         text_encoder, encode_fn, local_rank, global_rank, world_size, current_device):
    """Stage 3 DMD distillation. generator is already init (base+camera+cam_text+LoRA-G+HE), critic is built internally.

    real/fake share one critic base (toggle_lora_active off=real teacher / on=fake critic).
    """
    import time as _time
    from pipelines.wan.dmd_core import student_rollout, calc_dmd_loss, calc_critic_loss
    dtype = torch.bfloat16
    H_lat = args.num_height // 16
    W_lat = args.num_width // 16
    latent_shape = (48, H_lat, W_lat)
    M, K = args.dmd_num_blocks, args.dmd_block_K
    total_pix = (M * K - 1) * 4 + 1

    # 1. generator inherits Stage2 ckpt (LoRA-G + history_encoder)
    if getattr(args, 'generator_ckpt', None):
        _gtf = os.path.join(args.generator_ckpt, "diffusion_pytorch_model.safetensors")
        if os.path.exists(_gtf):
            echo_on_main_rank(f"[DMD] generator inherits Stage2 ckpt: {args.generator_ckpt}")
            _carry_stage1_into_stage2(generator, history_encoder, args.generator_ckpt, current_device)
        else:
            # Stage2 ckpt does not exist yet → temporarily use teacher (real_score_ckpt) to init generator base.
            #   Bare transformer key (X.weight) maps to LoRA-wrapped (X.base.weight); LoRA-G / history_encoder keep random init.
            _rs = getattr(args, 'real_score_ckpt', None)
            echo_on_main_rank(f"[DMD] ⚠ Stage2 ckpt does not exist({_gtf}), temporarily use teacher to init generator base: {_rs}")
            if _rs:
                _raw = safetensors.torch.load_file(_rs, device='cpu')
                _mk = set(generator.state_dict().keys())
                _mapped = {}
                for _k, _v in _raw.items():
                    if _k in _mk:
                        _mapped[_k] = _v
                    else:
                        _pp = _k.rsplit('.', 1)
                        if len(_pp) == 2 and f"{_pp[0]}.base.{_pp[1]}" in _mk:
                            _mapped[f"{_pp[0]}.base.{_pp[1]}"] = _v   # LowRankLinear: X.w → X.base.w
                _gm, _gu = generator.load_state_dict(_mapped, strict=False)
                echo_on_main_rank(f"[DMD] generator base <- teacher: mapped {len(_mapped)}/{len(_raw)} tensors, "
                                      f"missing={len(_gm)}, unexpected={len(_gu)}")

    # quantization-aware training (QAT) — generator nn.Linear fake-quant (★must be before FSDP wrap).
    #   NVFP4 (Blackwell/5090 FP4 kernel) or FP8 (Hopper/H200 FP8 kernel), choose one; after wrap the generator runs in
    #   quantized state (student) throughout, and the generator update adds a BF16↔quantized self-distillation KL (see below).
    _use_nvfp4 = bool(getattr(args, 'dmd_nvfp4', False))
    _use_fp8 = bool(getattr(args, 'dmd_fp8', False))
    if _use_nvfp4 and _use_fp8:
        raise ValueError("--dmd_nvfp4 and --dmd_fp8 are mutually exclusive, only one can be enabled (NVFP4=Blackwell, FP8=Hopper)")
    if _use_nvfp4:
        from pipelines.wan.dmd_core import enclose_generator_nvfp4
        _nvfp4_replaced, _nvfp4_cfg = enclose_generator_nvfp4(generator, args)
        echo_on_main_rank(
            f"[DMD-NVFP4] generator nn.Linear→NVFP4Linear: replaced {len(_nvfp4_replaced)} layers, "
            f"block={_nvfp4_cfg.block_size}, quant_act={_nvfp4_cfg.quantize_activations}, skip={_nvfp4_cfg.skip_modules}; "
            f"KL w={getattr(args,'nvfp4_kl_weight',0.03)} flow={getattr(args,'nvfp4_flow_weight',1.0)} "
            f"x0={getattr(args,'nvfp4_x0_weight',0.25)} warmup={getattr(args,'nvfp4_warmup_steps',0)}")
    if _use_fp8:
        from pipelines.wan.dmd_core import enclose_generator_fp8
        _fp8_replaced, _fp8_cfg = enclose_generator_fp8(generator, args)
        echo_on_main_rank(
            f"[DMD-FP8] generator nn.Linear→FP8Linear: replaced {len(_fp8_replaced)} layers, "
            f"quant_act={_fp8_cfg.quantize_activations}, skip={_fp8_cfg.skip_modules}; "
            f"KL w={getattr(args,'fp8_kl_weight',0.03)} flow={getattr(args,'fp8_flow_weight',1.0)} "
            f"x0={getattr(args,'fp8_x0_weight',0.25)} warmup={getattr(args,'fp8_warmup_steps',0)}")

    # 2. real_score (independent frozen teacher) + fake_score (independent critic, full-parameter online update)
    #    the two are independent, both initialized from the teacher ckpt, without LoRA / without memory module.
    #    real parameters are not updated (frozen); fake updates all parameters online.
    def _mk_score(tag):
        m = init_wan_transformer(args.pretrained_model_path, current_device, dtype,
                           model_type=getattr(args, 'wan_model_type', 'ti2v'), version=resolve_wan_version(args))
        _rs = getattr(args, 'real_score_ckpt', None)
        if _rs:
            _sd = safetensors.torch.load_file(_rs, device='cpu')
            _mm, _uu = m.load_state_dict(_sd, strict=False)
            echo_on_main_rank(f"[DMD] {tag} <- teacher {_rs}: {len(_sd)} tensors, "
                                  f"missing={len(_mm)}, unexpected={len(_uu)}")
        # only use discrete cam-text cross-attn, do not init continuous camera(control_adapter)/
        #   action_embedder, to save memory. cam-text only needs precompute_cam_text_embeddings.
        m = m.to(device=current_device, dtype=dtype)
        if getattr(args, 'use_camera_control', False):
            m.precompute_cam_text_embeddings(encode_fn, current_device, dtype)
        return m

    echo_on_main_rank("[DMD] Building real_score (frozen) + fake_score (full-parameter critic)...")
    real_score = _mk_score("real_score")
    for p in real_score.parameters():
        p.requires_grad = False
    real_score.eval()
    fake_score = _mk_score("fake_score")   # full-parameter trainable (no apply_lora, no memory)

    # --train_modules whitelist flexibly specifies generator trainable modules (qkv/patch_embedding/
    #   history_encoder/all). Compatible with old --gen_train_qkv_only (= train_modules=['qkv']).
    #   history_encoder is an independent module (not DiT), frozen/trained as a whole separately per whitelist. Must be set before FSDP wrap, otherwise it has no effect.
    #   Not passing train_modules and not passing gen_train_qkv_only → skip, keep default full-parameter (generator + HE both trainable).
    _train_modules = list(getattr(args, 'train_modules', None) or [])
    if not _train_modules and getattr(args, 'gen_train_qkv_only', False):
        _train_modules = ['qkv']
    if _train_modules:
        import re as _re
        _train_he = ('history_encoder' in _train_modules) or ('all' in _train_modules)
        _dit_mods = [m for m in _train_modules if m != 'history_encoder']
        # generator DiT parameter-name matching rules (None = unfreeze all)
        if 'all' in _dit_mods:
            _pats = None
        else:
            _pats = []
            if 'qkv' in _dit_mods:
                _pats.append(_re.compile(r'\.(self_attn|cross_attn)\.[qkv]\.(weight|bias)$'))
            if 'patch_embedding' in _dit_mods:
                _pats.append(_re.compile(r'(^|\.)patch_embedding\.'))
        def _dit_match(_n):
            return True if _pats is None else any(p.search(_n) for p in _pats)
        _n_train = _n_train_p = _n_freeze = _n_freeze_p = 0
        for _n, _p in generator.named_parameters():
            if _dit_match(_n):
                _p.requires_grad = True
                _n_train += 1; _n_train_p += _p.numel()
            else:
                _p.requires_grad = False
                _n_freeze += 1; _n_freeze_p += _p.numel()
        # history_encoder independent module: frozen/trained as a whole per whitelist (DMD has no HE for now → skip)
        _he_train_p = 0
        if history_encoder is not None:
            for _p in history_encoder.parameters():
                _p.requires_grad = bool(_train_he)
                if _train_he:
                    _he_train_p += _p.numel()
        echo_on_main_rank(
            f"[DMD-TrainModules] whitelist={_train_modules} → DiT unfrozen {_n_train} tensors/{_n_train_p/1e6:.1f}M, "
            f"frozen {_n_freeze} tensors/{_n_freeze_p/1e6:.1f}M; "
            f"history_encoder {'train' if _train_he else 'frozen'} ({_he_train_p/1e6:.1f}M).")

    # 3. FSDP wrap (generator + real_score + fake_score, 3 copies of 5B base)
    # generator/fake_score carry gradients → enable gradient_checkpointing to save activation memory;
    #   real_score is fully frozen + forward is no_grad throughout (calc_dmd_loss), no update and no checkpoint needed.
    _gc = getattr(args, 'gradient_checkpointing', False)
    generator = enclose_wan_model_with_fsdp(generator, args, current_device)
    fake_score = enclose_wan_model_with_fsdp(fake_score, args, current_device)
    args.gradient_checkpointing = False
    real_score = enclose_wan_model_with_fsdp(real_score, args, current_device)
    args.gradient_checkpointing = _gc

    # 4. Dual optimizer (generator: DiT requires_grad part + HE requires_grad part; critic: fake_score full-parameter)
    #   train_modules decides which of DiT/HE are trainable (qkv/patch_embedding/history_encoder/all); full-parameter if not passed.
    gen_params = [p for p in generator.parameters() if p.requires_grad]
    if history_encoder is not None:
        gen_params += [p for p in history_encoder.parameters() if p.requires_grad]
    critic_params = [p for p in fake_score.parameters() if p.requires_grad]
    echo_on_main_rank(f"[DMD] generator trainable={sum(p.numel() for p in gen_params)/1e6:.1f}M, "
                          f"fake_score(critic) trainable={sum(p.numel() for p in critic_params)/1e6:.1f}M")
    # betas follow the Self-Forcing DMD setting (beta1=0.0, beta2=0.999;
    #   both gen and critic). beta1=0 = no first-order momentum, a typical DMD setting (more responsive to fast-changing scores).
    gen_opt = torch.optim.AdamW(gen_params, lr=args.dmd_generator_lr, weight_decay=0.0, betas=(0.0, 0.999))
    critic_opt = torch.optim.AdamW(critic_params, lr=args.dmd_critic_lr, weight_decay=0.0, betas=(0.0, 0.999))

    # 4b. best-file trick (all opt-in): GAN discriminator + EMA. (GT-latent regression uses vae encoding inside the loop)
    _use_gan = getattr(args, 'dmd_use_gan', False)
    _use_gt_reg = getattr(args, 'dmd_use_gt_reg', False)
    _gt_reg_w = float(getattr(args, 'dmd_gt_reg_weight', 0.0))
    _gan_w = float(getattr(args, 'dmd_gan_weight', 0.0))
    # SFT + forward-KL anchoring (port from yume) — resist DMD collapse/mode-shrinkage, preserve long videos and motion
    _sft_w = float(getattr(args, 'dmd_sft_weight', 0.0))
    _rfkl_w = float(getattr(args, 'dmd_real_fkl_weight', 0.0))
    _fkl_w = float(getattr(args, 'dmd_fkl_weight', 0.0))
    _use_sft = _sft_w > 0.0
    _use_real_fkl = _rfkl_w > 0.0
    _use_fkl = _fkl_w > 0.0
    discriminator = None
    disc_opt = None
    _disc_params = []
    if _use_gan:
        from ADD.models.discriminator import ProjectedDiscriminator
        # the discriminator uses [plain module + manual all-reduce of gradients], not wrapped in DDP.
        #   Reason: if wrapped in DDP, the generator's GAN term needs to use .module forward (bypassing DDP.forward), and at backward
        #   DDP's param hook has not prepared the reducer → deadlock/error. plain + manual sync is most stable, and also convenient for the gen term to reuse the same module.
        discriminator = ProjectedDiscriminator(c_dim=384).to(current_device)   # internal DINO is already frozen
        _disc_params = [p for p in discriminator.parameters() if p.requires_grad]
        # discriminator uses an independent LR(--dmd_disc_lr, default 1e-4), no longer wrongly using critic's 2e-6
        #   (2e-6 too low → discriminator cannot learn, disc loss stuck at 2.0, GAN ineffective and cannot prevent collapse).
        _disc_lr = float(getattr(args, 'dmd_disc_lr', 1e-4) or 1e-4)
        disc_opt = torch.optim.AdamW(_disc_params, lr=_disc_lr, weight_decay=0.0, betas=(0.0, 0.999))
        echo_on_main_rank(f"[DMD-GAN] discriminator LR = {_disc_lr:.1e} (independent of critic_lr={args.dmd_critic_lr:.1e})")
        echo_on_main_rank(f"[DMD-GAN] ProjectedDiscriminator built (plain+manual allreduce): trainable params "
                              f"{sum(p.numel() for p in _disc_params)/1e6:.1f}M, gan_weight={_gan_w}")
    ema = None
    if getattr(args, 'dmd_use_ema', False):
        from ADD.ldm.modules.ema import LitEma
        ema = LitEma(generator, decay=float(getattr(args, 'dmd_ema_decay', 0.999)))
        ema.to(current_device)
        echo_on_main_rank(f"[DMD-EMA] generator EMA built: decay={getattr(args,'dmd_ema_decay',0.999)}")

    # 5. dataloader + neg prompt
    # 18: BUGFIX — the neg prompt previously used getattr(args,'negative_prompt',''),
    #   but train_stage2.py's argparse never defines --negative_prompt → it was actually an empty string.
    #   CFG=5 with an empty neg prompt → v_u+5*(v_c-v_u) extrapolates severely → teacher 50step video overexposed/broken.
    #   Switched to the official long neg prompt WAN_NEG_PROMPT_TEXT (consistent with normal validation line 3040/3066).
    train_dataloader, train_sampler, _ = build_train_loader(args)
    with torch.no_grad():
        _neg_text = WAN_NEG_PROMPT_TEXT
        neg_emb = encode_fn([_neg_text])[0].to(current_device, dtype).unsqueeze(0)
    echo_on_main_rank(f"[DMD] neg_emb uses official WAN_NEG_PROMPT_TEXT, tokens={neg_emb.shape[1]}, "
                          f"text[:30]={_neg_text[:30]!r}")

    # text can be offloaded to CPU to save memory (precompute cam-text + neg already done on GPU;
    #   afterwards each step's caption encode goes through CPU — encode_fn auto-adapts using text_encoder.device, slow but a small fraction).
    #   ★ Models (generator/real/fake) do NOT do cpu offload (DMD sh does not pass --use_cpu_offload, FSDP all on GPU).
    if getattr(args, 'offload_text_encoder', False) and text_encoder is not None:
        if hasattr(text_encoder, 'model'):
            text_encoder.model.to('cpu')
        elif hasattr(text_encoder, 'to'):
            text_encoder.to('cpu')
        torch.cuda.empty_cache()
        echo_on_main_rank("[DMD] text_encoder offloaded to CPU (caption encode goes through CPU)")

    ratio = args.dfake_gen_update_ratio
    # gradient accumulation — critic accumulates _accum consecutive micro-steps, generator accumulates _accum gen-updates then steps.
    #   (DMD has batch=1 per step and the model forces B=1, so a larger effective batch can only come from grad accum; smooths DMD noisy gradients → more stable).
    #   global_step is still counted per micro-step (per batch); warmup/validation/checkpoint continue to use global_step unchanged.
    _accum = max(1, int(getattr(args, 'gradient_accumulation_steps', 1)))
    _gen_update_count = 0
    if _accum > 1:
        echo_on_main_rank(f"[DMD] gradient accumulation grad_accum={_accum} (critic steps once every {_accum} micro-steps, gen steps once every {_accum} gen-updates)")
    _critic_warmup = getattr(args, 'dmd_critic_warmup_steps', 0)   # ★ first N steps train only critic
    if _critic_warmup > 0:
        echo_on_main_rank(f"[DMD] critic warmup: first {_critic_warmup} steps update only critic, then enable generator update")
    save_steps = getattr(args, 'checkpointing_steps', 500)
    _val_interval = getattr(args, 'validation_interval', 200)
    _first_val = getattr(args, 'first_validation_step', 0)
    total_steps = getattr(args, 'max_train_steps', 0) or 100000
    generator.train(); fake_score.train(); real_score.eval()
    if history_encoder is not None:
        history_encoder.train()

    global_step = 0
    for epoch in range(10 ** 9):
        train_sampler.set_epoch(epoch)
        for data_item in train_dataloader:
            if global_step >= total_steps:
                break
            if isinstance(data_item, dict) and data_item.get("skip"):
                continue
            _t0 = _time.time()
            cap, K_ctrl, c2w, c2w_raw, has_cam, ph, pw, _biwm_action_labels = _dmd_extract_cam(data_item)
            with torch.no_grad():
                # when text_encoder is offloaded to CPU, encode_fn's device is the cuda captured
                #   by the closure (it would put ids on cuda while the model is on cpu → mismatch). So during encode temporarily
                #   move text_encoder to GPU, then move back to CPU after encode (encode is before the 3×5B forward, does not hit the memory peak).
                _te_off = getattr(args, 'offload_text_encoder', False) and text_encoder is not None
                if _te_off:
                    getattr(text_encoder, 'model', text_encoder).to(current_device)
                cap_emb = encode_fn([cap])[0].to(current_device, dtype).unsqueeze(0)
                if _te_off:
                    getattr(text_encoder, 'model', text_encoder).to('cpu')
                    torch.cuda.empty_cache()

            # discrete camera control — full_labels [1, M*K] full-segment latent-level action_label.
            #   BiwmCamCaptionData directly gives precomputed action_labels (action_frames parsing, consistent with stage1);
            #   passed to student_rollout (sliced per block) / teacher(cond) / critic / dmd_loss / validation(joystick).
            #   Non-biWM datasets (multi-source) are still pure t2v (action_labels=None).
            full_labels = None
            if getattr(args, 'use_camera_control', False) and _biwm_action_labels is not None:
                _MK = int(args.dmd_num_blocks) * int(args.dmd_block_K)
                _al = _biwm_action_labels
                if not isinstance(_al, torch.Tensor):
                    _al = torch.as_tensor(_al)
                if _al.dim() == 1:
                    _al = _al.unsqueeze(0)                    # [1, L]
                _al = _al.to(device=current_device, dtype=torch.long)
                # Align to the rollout total latent count M*K (biWM 77 frames→20 latent, default M*K=20 is exactly equal)
                if _al.shape[1] != _MK:
                    _al = F.interpolate(_al.float().unsqueeze(1), size=_MK,
                                        mode='nearest').squeeze(1).long()
                full_labels = _al

            # GT latent (for GAN real / GT regression): di[0] real video → VAE latent. None if unused.
            gt_latent = None
            # SFT/real-FKL also need gt_latent (complete real video, full length dmd_num_blocks*K=20)
            if (_use_gan or _use_gt_reg or _use_sft or _use_real_fkl) and vae_encoder is not None:
                gt_latent = _dmd_embed_gt_latent(data_item["pixel_values"], vae_encoder, current_device,
                                                  int(args.dmd_num_blocks) * int(args.dmd_block_K),
                                                  expect_chw=latent_shape)

            # block-count curriculum — first dmd_block_warmup_steps steps fix single-block cold start (pure t2v, no history),
            #   afterwards num_blocks=None → sample per dmd_block_counts/probs (or dmd_num_blocks). critic/gen use the same M at the same step.
            _warm_steps = int(getattr(args, 'dmd_block_warmup_steps', 0))
            if global_step < _warm_steps:
                _step_num_blocks = int(getattr(args, 'dmd_block_warmup_count', 1))
            else:
                # after warmup, mixed-sample M∈dmd_block_counts; [sample once] pass to critic+gen to guarantee same M at the same step
                #   (otherwise each samples a different M → the score critic gives and the distribution gen produces do not match, more unstable).
                from pipelines.wan.dmd_core import _draw_dmd_block_count
                _step_num_blocks = _draw_dmd_block_count(args, current_device)

            # --- critic step (every step): fake_score full-parameter learns to denoise generator output ---
            # gradient accumulation — accumulate every _accum consecutive micro-steps (different batches) then step.
            _crit_micro = global_step % _accum
            if _crit_micro == 0:
                critic_opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                x0_c, _, _dto_c, _M_c = student_rollout(generator, history_encoder, args, cap_emb,
                                                          full_labels, latent_shape, current_device, dtype,
                                                          num_blocks=_step_num_blocks)
            closs, _ = calc_critic_loss(fake_score, x0_c, cap_emb, full_labels, args, current_device,
                                           denoised_to=_dto_c)
            (closs / _accum).backward()                          # accumulate: scale loss by 1/N
            _cgn = float('nan')
            if _crit_micro == _accum - 1:                        # accumulated N → clip + step
                # critic gradient (total norm before clip, FSDP local shard; check whether nonzero/stable)
                _cgn = float(torch.nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm))
                critic_opt.step()

            # --- discriminator step (GAN, real=gt_latent, fake=generator x0_c) ---
            #   deadlock fix: disc's backward triggers a DDP all-reduce (collective), which must be executed
            #     consistently by all ranks —— but whether gt_latent is available is [per-rank] (each rank's video resolution/frame count differs, None if unavailable).
            #     If some ranks have gt and run disc while others skip → all-reduce misalignment → NCCL silent deadlock (this is exactly the step1 stall).
            #     Use all_reduce(MIN) for global consistency: only run disc when [all ranks have gt], otherwise everyone skips.
            _dloss = float('nan')
            if discriminator is not None:
                _has_gt = 1 if gt_latent is not None else 0
                if dist.is_initialized() and dist.get_world_size() > 1:
                    _gt_flag = torch.tensor([_has_gt], device=current_device, dtype=torch.int32)
                    dist.all_reduce(_gt_flag, op=dist.ReduceOp.MIN)
                    _all_have_gt = bool(_gt_flag.item())
                else:
                    _all_have_gt = bool(_has_gt)
                if _all_have_gt:
                    from pipelines.wan.dmd_core import calc_discriminator_loss
                    disc_opt.zero_grad(set_to_none=True)
                    _Fc = min(x0_c.shape[2], gt_latent.shape[2])
                    with torch.autocast('cuda', dtype=dtype):
                        dloss = calc_discriminator_loss(discriminator, x0_c[:, :, :_Fc], gt_latent[:, :, :_Fc])
                    dloss.backward()
                    # ★ Manually sync discriminator gradients (plain module, average across ranks)
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        _ws = dist.get_world_size()
                        for _p in _disc_params:
                            if _p.grad is not None:
                                dist.all_reduce(_p.grad, op=dist.ReduceOp.SUM)
                                _p.grad /= _ws
                    torch.nn.utils.clip_grad_norm_(_disc_params, args.max_grad_norm)
                    disc_opt.step()
                    _dloss = float(dloss.item())

            # --- generator step (every ratio steps): DMD loss (real frozen + fake independent) ---
            gloss_val = float('nan')
            _ggn = _g_lora = _g_he = _g_patch = float('nan')
            _l_sft = _l_fkl = _l_rfkl = float('nan')   # ★ SFT/forward-KL log (init at outer scope, defined even on non-gen steps)
            _M_g = -1
            if global_step % ratio == 0 and global_step >= _critic_warmup:
                _gen_micro = _gen_update_count % _accum          # ★ gradient accumulation: gen accumulates _accum gen-updates then steps
                if _gen_micro == 0:
                    gen_opt.zero_grad(set_to_none=True)
                x0_g, gmask, _dto_g, _M_g = student_rollout(generator, history_encoder, args, cap_emb,
                                                              full_labels, latent_shape, current_device, dtype,
                                                              num_blocks=_step_num_blocks)
                _gt_for_reg = (gt_latent[:, :, :x0_g.shape[2]]
                               if (_use_gt_reg and gt_latent is not None and gt_latent.shape[2] >= x0_g.shape[2])
                               else None)
                gloss, _gi = calc_dmd_loss(real_score, fake_score, x0_g, cap_emb, neg_emb,
                                              full_labels, args, current_device, gradient_mask=gmask,
                                              denoised_to=_dto_g,
                                              gt_latent=_gt_for_reg, gt_reg_weight=_gt_reg_w)
                # ★ GAN term (plain discriminator, gradient flows only to generator; the discriminator's local gradient is cleared by zero_grad on the next disc step)
                if discriminator is not None:
                    from pipelines.wan.dmd_core import calc_generator_gan_loss
                    with torch.autocast('cuda', dtype=dtype):
                        _ganl = calc_generator_gan_loss(discriminator, x0_g)
                    gloss = gloss + _gan_w * _ganl.to(gloss.dtype)
                # SFT + forward-KL anchoring (port from yume) — resist collapse/mode-shrinkage, preserve long videos and motion.
                #   SFT/real-FKL use [complete real video length] (gt_latent, 20 frames), decoupled from the dynamic-M rollout (even warmup M=1 learns full length).
                _l_sft = _l_fkl = _l_rfkl = float('nan')
                if _use_sft and gt_latent is not None:
                    from pipelines.wan.dmd_core import calc_sft_loss
                    _sft = calc_sft_loss(generator, gt_latent, cap_emb, full_labels, args, current_device)
                    gloss = gloss + _sft_w * _sft.to(gloss.dtype)
                    _l_sft = float(_sft.detach())
                if _use_real_fkl and gt_latent is not None:
                    from pipelines.wan.dmd_core import calc_real_fkl_loss
                    _rfkl = calc_real_fkl_loss(generator, gt_latent, cap_emb, full_labels, args, current_device)
                    gloss = gloss + _rfkl_w * _rfkl.to(gloss.dtype)
                    _l_rfkl = float(_rfkl.detach())
                if _use_fkl:
                    from pipelines.wan.dmd_core import calc_teacher_fkl_loss
                    _Tg = int(args.dmd_num_blocks) * int(args.dmd_block_K)   # full generation length 20
                    _fkl = calc_teacher_fkl_loss(generator, real_score, cap_emb, neg_emb, full_labels,
                                                    latent_shape, _Tg, args, current_device, dtype)
                    gloss = gloss + _fkl_w * _fkl.to(gloss.dtype)
                    _l_fkl = float(_fkl.detach())
                # quantization self-distillation KL —— student(quantized) aligns with BF16(itself) output, resists quantization quality drop.
                _nvfp4_kl_val = float('nan')
                if _use_nvfp4 and global_step >= getattr(args, 'nvfp4_warmup_steps', 0)\
                        and getattr(args, 'nvfp4_kl_weight', 0.0) > 0:
                    from pipelines.wan.dmd_core import calc_nvfp4_kl
                    _nvfp4_kl, _nvfp4_log = calc_nvfp4_kl(generator, x0_g, cap_emb, full_labels,
                                                             args, current_device)
                    gloss = gloss + float(args.nvfp4_kl_weight) * _nvfp4_kl.to(gloss.dtype)
                    _nvfp4_kl_val = float(_nvfp4_kl.detach())
                if _use_fp8 and global_step >= getattr(args, 'fp8_warmup_steps', 0)\
                        and getattr(args, 'fp8_kl_weight', 0.0) > 0:
                    from pipelines.wan.dmd_core import calc_fp8_kl
                    _fp8_kl, _fp8_log = calc_fp8_kl(generator, x0_g, cap_emb, full_labels,
                                                       args, current_device)
                    gloss = gloss + float(args.fp8_kl_weight) * _fp8_kl.to(gloss.dtype)
                    _nvfp4_kl_val = float(_fp8_kl.detach())   # reuse the same print field
                (gloss / _accum).backward()                      # ★ accumulate: scale loss by 1/N
                gloss_val = gloss.item()
                if _gen_micro == _accum - 1:                      # accumulated N gen-updates → clip + step + ema
                    # ★ generator grouped gradients (before clip): LoRA-G / history_encoder / patch_embedding —— confirm each part is learning
                    _g_lora = _dmd_gradient_norm([p for n, p in generator.named_parameters()
                                              if p.requires_grad and ('lora_A' in n or 'lora_B' in n)])
                    _g_patch = _dmd_gradient_norm([p for n, p in generator.named_parameters()
                                               if p.requires_grad and 'patch_embedding' in n])
                    _g_he = (_dmd_gradient_norm([p for p in history_encoder.parameters() if p.requires_grad])
                             if history_encoder is not None else float('nan'))
                    _ggn = float(torch.nn.utils.clip_grad_norm_(gen_params, args.max_grad_norm))
                    gen_opt.step()
                    if ema is not None:    # ★ EMA: update shadow weights after each generator update (disable EMA on failure without interrupting training)
                        try:
                            ema(generator)
                        except Exception as _ee:
                            echo_on_main_rank(f"[DMD-EMA] update failed, disabling EMA: {_ee}")
                            ema = None
                _gen_update_count += 1                            # count each gen-update (used for accumulation grouping)

            # the print covers both "every 10 steps" and "every generator update step" (when ratio and 10 are misaligned,
            #   the old code printed the placeholder nan gloss_val on non-update steps, mistaken for divergence). Update steps print the real gen loss, otherwise marked as not updated.
            _is_gen_step = (global_step % ratio == 0 and global_step >= _critic_warmup)
            if global_rank == 0 and (global_step % 10 == 0 or _is_gen_step):
                if _is_gen_step:
                    _gmsg = (f"gen={gloss_val:.4f} | gen_grad={_ggn:.3e} "
                             f"(lora={_g_lora:.3e} he={_g_he:.3e} patch={_g_patch:.3e}) genM={_M_g}")
                elif global_step < _critic_warmup:
                    _gmsg = f"gen=(critic warmup {global_step}/{_critic_warmup}, not updating gen yet)"
                else:
                    _gmsg = "gen=(this step is not a generator update step)"
                # dynamic DMD — print this step's total block count M for critic/gen rollout (history=M-1)
                _extra = ""
                if discriminator is not None:
                    _extra += f" disc={_dloss:.4f}"
                if _use_gt_reg and _is_gen_step and isinstance(_gi, dict) and 'gt_reg' in _gi:
                    _extra += f" gt_reg={_gi['gt_reg']:.4f}"
                if (_use_nvfp4 or _use_fp8) and _is_gen_step and not math.isnan(_nvfp4_kl_val):
                    _extra += f" {'fp8' if _use_fp8 else 'nvfp4'}_kl={_nvfp4_kl_val:.4f}"
                if _is_gen_step:   # ★ SFT / forward-KL anchoring terms
                    if not math.isnan(_l_sft):  _extra += f" sft={_l_sft:.4f}"
                    if not math.isnan(_l_fkl):  _extra += f" fkl={_l_fkl:.4f}"
                    if not math.isnan(_l_rfkl): _extra += f" rfkl={_l_rfkl:.4f}"
                print(f"[DMD step={global_step}] critic={closs.item():.4f}(grad={_cgn:.3e},criticM={_M_c}) "
                      f"{_gmsg}{_extra} ({_time.time()-_t0:.1f}s)", flush=True)

            # --- validation: periodically use generator 4-step to generate the full segment (check distillation effect + camera joystick) ---
            if (vae_decoder is not None and _val_interval > 0
                    and (global_step % _val_interval == 0 or global_step == _first_val)):
                # ★ EMA: validation temporarily swaps in the EMA shadow weights to see the effect, then restores afterwards (fall back to live weights on failure)
                _ema_applied = False
                if ema is not None:
                    try:
                        ema.store(generator.parameters()); ema.copy_to(generator); _ema_applied = True
                    except Exception as _ee:
                        echo_on_main_rank(f"[DMD-EMA] copy_to failed, validation uses live weights: {_ee}")
                _dmd_evaluate(generator, history_encoder, real_score, vae_decoder, args, cap_emb,
                              neg_emb, full_labels, latent_shape, current_device, dtype,
                              args.output_dir, global_step, global_rank, getattr(args, 'fps', 24.0),
                              caption_text=cap)
                if _ema_applied:
                    try:
                        ema.restore(generator.parameters())
                    except Exception as _ee:
                        echo_on_main_rank(f"[DMD-EMA] restore failed: {_ee}")

            if global_step > 0 and global_step % save_steps == 0:
                try:
                    store_checkpoint(generator, global_rank, args.output_dir, global_step)
                    _store_history_encoder(history_encoder, global_rank, args.output_dir, global_step)
                    dist.barrier()
                    echo_on_main_rank(f"[DMD] saved generator ckpt step={global_step}")
                except Exception as _se:
                    print(f"[Rank {global_rank}] DMD ckpt save failed: {_se}", flush=True)
            # clear memory fragmentation each step (3 5B models + critic/gen alternation easily fragments memory)
            torch.cuda.empty_cache()
            global_step += 1
        if global_step >= total_steps:
            break

    store_checkpoint(generator, global_rank, args.output_dir, global_step)
    _store_history_encoder(history_encoder, global_rank, args.output_dir, global_step)
    echo_on_main_rank("[DMD] distillation complete")


# =============================================================================
# Main function
# =============================================================================
def main(args: argparse.Namespace) -> None:
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
    # NCCL timeout 36000s(10h) → 1800s(30min)
    # Previously a GPFS IO hang stalled one rank on data, while other ranks waited a full 10h on broadcast before recovering.
    # Reduced to 30min: on storage jitter, at most 30min is wasted before fail-restart, instead of waiting 10h in vain.
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
    gloo_timeout = datetime.timedelta(seconds=1800)   # 36000→1800 (same as NCCL)
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

    # Stage3 DMD distillation — independent branch (transformer is the generator), does not go through the rollout flow below
    if getattr(args, 'dmd_distill', False):
        echo_on_main_rank("[DMD] === Stage 3 Distribution Matching Distillation ===")
        execute_dmd_distillation(args, transformer, history_encoder, vae_encoder, vae_decoder,
                             text_encoder, encode_fn, local_rank, global_rank, world_size, current_device)
        if dist.is_initialized():
            dist.barrier()
        return

    # ★ Stage 2 inherits Stage 1: load LoRA-wrapped transformer state + history_encoder
    # Must be before FSDP wrap (at this point transformer has apply_lora, keys contain .lora_A/.lora_B to match the Stage1 ckpt)
    _stage1_ckpt = getattr(args, 'stage1_ckpt', None)
    if getattr(args, 'rollout_pretrain', False) and _stage1_ckpt:
        echo_on_main_rank(f"[Stage2] inherit Stage 1 ckpt: {_stage1_ckpt}")
        _carry_stage1_into_stage2(transformer, history_encoder, _stage1_ckpt, current_device)

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

    # === Caption Refine VLM ===
        dist.barrier()

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
    # packforcing_pretrain/rollout mode must add history_encoder parameters into the optimizer
    # otherwise HE parameter grads are computed but not updated → HR abs never changes (HR encoder does not learn)
    _extra_params = None
    if history_encoder is not None:
        _he_params = [p for p in history_encoder.parameters() if p.requires_grad]
        _extra_params = _he_params
        echo_on_main_rank(f"[Optimizer] Adding history_encoder parameters ({sum(p.numel() for p in _he_params)/1e6:.2f}M) to optimizer trainable")
    optimizer, lr_scheduler = build_optimizer_and_scheduler(transformer, args, extra_params=_extra_params)

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

    # === Adversarial Loss ===
    _gan_weight = getattr(args, 'gan_weight', 0.01)
    if getattr(args, 'distill', False) and _gan_weight > 0:
        from pipelines.wan.adversarial_loss import GanCoach
        adv_trainer = GanCoach(
            device=current_device,
            fsdp_kwargs=None,
            lr=getattr(args, 'discriminator_learning_rate', 1e-4),
            gan_weight=_gan_weight,
            latent_channels=wc.WAN_LATENT_CHANNEL_COUNT,
        )
        transformer._adv_trainer = adv_trainer
        echo_on_main_rank(f"[Distill] Adversarial training: weight={_gan_weight}")
    else:
        transformer._adv_trainer = None

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
            try:
                data_item = next(_dataloader_iter)
            except StopIteration:
                break
            except Exception as _dl_err:
                print(f"[Rank {global_rank}] DataLoader error: {_dl_err}", flush=True)
                continue

            if global_step >= total_train_steps:
                break

            step_start_time = time.time()

            # Set current step for camera warmup + packforcing_pretrain Mem ratio print
            # packforcing_pretrain mode sets _current_train_step on the WanModel top level,
            # so the [FrameQuery-Mem step=N] print shows the real step instead of 0
            _relative_step = global_step - resumed_step
            for _blk in transformer.modules():
                if hasattr(_blk, '_current_train_step'):
                    _blk._current_train_step = _relative_step
            # After FSDP wrap, the transformer top level is FSDP, but the inner self._fsdp_wrapped_module is the real WanModel
            _root = transformer
            if hasattr(_root, '_fsdp_wrapped_module'):
                _root = _root._fsdp_wrapped_module
            if hasattr(_root, 'module'):
                _root = _root.module
            _root._current_train_step = global_step

            # Regular DMD training step (PackForcing pretrain / rollout is held out of this release).
            if getattr(args, 'packforcing_pretrain', False) or getattr(args, 'rollout_pretrain', False):
                raise RuntimeError("PackForcing pretrain / rollout training is not available in this release.")
            else:
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
                try:
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
                except Exception as tb_err:
                    if global_step % 100 == 0:
                        echo_on_main_rank(f"[TensorBoard] Write failed: {tb_err}")

            # === Validation ===
            # first_validation_step also takes effect on cold start
            # - Cold start (resumed_step=0): first val at global_step == first_validation_step,
            #   then once every EVAL_INTERVAL steps
            # - Resume (resumed_step>0): first val at global_step == resumed_step + first_val_step
            _first_val_step = getattr(args, 'first_validation_step', 0)
            _resume_buffer = _first_val_step if (_first_val_step > 0 and resumed_step > 0) else 0
            _should_validate = (global_step % EVAL_INTERVAL == 0 and global_step > 0
                                and global_step != resumed_step
                                and global_step >= resumed_step + _resume_buffer)
            if _first_val_step > 0:
                if resumed_step == 0 and global_step == _first_val_step:
                    _should_validate = True   # cold start first val
                elif resumed_step > 0 and global_step == resumed_step + _first_val_step:
                    _should_validate = True   # first val after resume buffer
            if validation_only_mode or _should_validate:
                echo_on_main_rank(f"\n{'='*60}")
                echo_on_main_rank(f"[Validation] Step {global_step}")
                echo_on_main_rank(f"{'='*60}")

                free_gpu_memory()

                # ★ packforcing_pretrain / rollout modes go through their own dedicated validation
                if getattr(args, 'packforcing_pretrain', False) or getattr(args, 'rollout_pretrain', False):
                    try:
                        raise RuntimeError("PackForcing pretrain / rollout validation is not available in this release.")
                    except Exception as _val_err:
                        import traceback
                        echo_on_main_rank(f"[Validation] packforcing_pretrain/rollout validation failed: {_val_err}")
                        traceback.print_exc()
                    # bug fix: save must not be skipped by validation's continue!
                    # If checkpointing_steps is a multiple of validation_interval (e.g. 100/50),
                    # every ckpt step coincides with validation, and continue would make the checkpoint never save.
                    # So check save before continue.
                    if global_step % args.checkpointing_steps == 0 and global_step > 0:
                        echo_on_main_rank(f"[Checkpoint] Saving step={global_step} (validation step)")
                        try:
                            free_gpu_memory()
                            store_checkpoint(transformer, global_rank, args.output_dir, global_step)
                            _store_history_encoder(history_encoder, global_rank, args.output_dir, global_step)
                            dist.barrier()
                            echo_on_main_rank(f"[Checkpoint] Saved step={global_step}")
                        except Exception as _ce:
                            print(f"[Rank {global_rank}] Checkpoint save failed: {_ce}", flush=True)
                            import traceback
                            traceback.print_exc()
                    # after validation, skip t2v sampling; global_step += 1 before continue (otherwise infinite loop)
                    global_step += 1
                    continue

                validation_prompt = last_prompt if last_prompt else "A beautiful sunset over the ocean."
                val_height = getattr(args, 'validation_height', args.num_height)
                val_width = getattr(args, 'validation_width', args.num_width)

                # Validation frames: use GT video's actual frame count (not fixed param)
                # This ensures Plucker, action_labels, GT latent all have matching frame counts
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

                try:
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
                        stg_scale=getattr(args, 'stg_scale', 0.0),
                        stg_blocks=[int(b) for b in getattr(args, 'stg_blocks', '').split(',') if b.strip()] if getattr(args, 'stg_blocks', '') else None,
                        rescale_scale=getattr(args, 'rescale_scale', 0.7),
                        source=str(last_source) if last_source else "unknown",
                        gt_pixel_frames=last_gt_pixel_frames,
                    )
                except Exception as e:
                    print(f"[Rank {global_rank}] Validation failed: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    dist.barrier()

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
                try:
                    free_gpu_memory()
                    store_checkpoint(transformer, global_rank, args.output_dir, global_step)
                    _store_history_encoder(history_encoder, global_rank, args.output_dir, global_step)
                    dist.barrier()
                    echo_on_main_rank(f"[Checkpoint] Saved step={global_step}")
                except Exception as e:
                    print(f"[Rank {global_rank}] Checkpoint save failed: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

            free_gpu_memory()
            global_step += 1

        if global_step >= total_train_steps:
            echo_on_main_rank(f"[Epoch {epoch}] Reached total steps {total_train_steps}")
            break

    # Final checkpoint
    store_checkpoint(transformer, global_rank, args.output_dir, global_step)
    _store_history_encoder(history_encoder, global_rank, args.output_dir, global_step)

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
                        help="Wan backbone version: 'auto' detects from model path, or force '2.1'(1.3B)/'2.2'(5B).")

    # ===== Dataset =====
    parser.add_argument("--num_height", type=int, default=544)
    parser.add_argument("--num_width", type=int, default=960)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2,
                       help="Number of batches prefetched per worker, total prefetch = num_workers × prefetch_factor")

    # ===== BiWM discrete camera-text dataset (dataset/videos, pose parsed from json into latent-level discrete actions) =====
    parser.add_argument("--use_biwm_camera_dataset", action="store_true",
                        help="Use BiwmCamCaptionData (dataset/videos), action_labels given directly by action_frames (DMD full_labels also uses it)")
    parser.add_argument("--biwm_video_dir", type=str, default=None,
                        help="dataset/videos directory (each subdirectory contains gen.mp4)")
    parser.add_argument("--biwm_caption_json", type=str, default=None,
                        help="videos_syn.json (list, index==6-digit id, contains caption + action_frames)")

    # ===== Training config =====
    parser.add_argument("--stage1_ckpt", type=str, default=None,
                       help="★ Stage 2 inherits Stage 1's checkpoint-N directory (load LoRA + history_encoder)")
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
    parser.add_argument("--camera_injection_mode", type=str, default="wan_inject",
                       choices=["scale_shift", "additive", "wan_inject"])
    parser.add_argument("--num_actions", type=int, default=81)
    parser.add_argument("--no_discrete_camera", action="store_true")
    parser.add_argument("--absolute_rot_thresh", action="store_true")
    parser.add_argument("--rot_thresh_min_deg", type=float, default=0.5)
    parser.add_argument("--train_sigma_shift", type=float, default=0.0,
                        help="Shift applied to the TRAINING sigma: shifted = s*σ/(1+(s-1)*σ). 0=uniform, >0=bias toward high-noise.")
    parser.add_argument("--val_sigma_shift", type=float, default=5.0,
                        help="Shift applied to the VALIDATION/sampling sigma schedule (Wan default ~3-5).")
    parser.add_argument("--freeze_dit_weights", action="store_true")

    # ===== trainable_modules =====
    parser.add_argument("--trainable_modules", type=str, default=None)

    # ===== Adversarial distillation =====
    parser.add_argument("--distill", action="store_true", default=False)
    parser.add_argument("--discriminator_learning_rate", type=float, default=1e-4)
    parser.add_argument("--gan_weight", type=float, default=0.01)

    # ===== Multi-source data =====

    # ===== FSDP =====
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="full")
    parser.add_argument("--use_cpu_offload", action="store_true")
    parser.add_argument("--master_weight_type", type=str, default="bf16", choices=["fp32", "bf16"])

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
    parser.add_argument("--stg_scale", type=float, default=0.0)
    parser.add_argument("--stg_blocks", type=str, default="")
    parser.add_argument("--rescale_scale", type=float, default=0.7)

    # ===== Output =====
    parser.add_argument("--output_dir", type=str, default="./outputs_wan_22")
    parser.add_argument("--checkpointing_steps", type=int, default=500)

    # ===== Parallel =====
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--cp_size", type=int, default=1)

    # ===== Memory optimization =====
    parser.add_argument("--offload_text_encoder", action="store_true")
    parser.add_argument("--torch_compile", action="store_true", help="torch.compile VAE for memory savings")
    parser.add_argument("--vae_tiling", action="store_true")
    parser.add_argument("--enable_memory_efficient_attention", action="store_true")

    # ===== Caption Refine =====

    # ===== Debug =====
    parser.add_argument("--debug", type=lambda x: x.lower() == 'true', default=False)

    from pipelines.wan.train_stage1 import register_lora_args
    register_lora_args(parser)

    # Stage3 DMD distillation args
    try:
        from pipelines.wan.dmd_core import register_dmd_args
        register_dmd_args(parser)
    except ImportError as _e:
        print(f"[train_stage2] WARN: pipelines.wan.dmd_core unavailable ({_e}), skipping dmd args injection")

    args = parser.parse_args()
    return args


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    args = parse_cli_options()
    main(args)
