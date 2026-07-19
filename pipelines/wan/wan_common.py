# Shared Wan2.1/2.2 backbone helpers — defined once, imported by train_stage1 / train_stage2 /
# infer_stage1 / infer_stage2 (VAE / transformer / text-encoder / FSDP / optimizer / checkpoint /
# dataloader / video IO / LoRA / Wan version resolution).


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


# ===== shared constants =====

WAN_VAE_TIME_STRIDE = 4    # temporal compression 4x

WAN_VAE_SPACE_STRIDE = 16    # spatial compression 16x

WAN_LATENT_CHANNEL_COUNT = 48       # Wan2.2 VAE latent channel count (z_dim=48)

WAN_VAE_CHANNEL_DIM = 160            # Wan2.2 VAE encoder first-layer dimension

WAN_TEXT_LENGTH = 512              # text sequence length

WAN_PATCH_DIM = (1, 2, 2)     # Patch size

WAN_NEG_PROMPT_TEXT = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

WAN_TI2V_5B_SETTINGS = {
    'model_type': 'ti2v',
    'dim': 3072,
    'ffn_dim': 14336,
    'freq_dim': 256,
    'text_dim': 4096,
    'in_dim': 48,
    'out_dim': 48,
    'num_heads': 24,
    'num_layers': 30,
    'patch_size': (1, 2, 2),
    'text_len': 512,
    'qk_norm': True,
    'cross_attn_norm': True,
}

WAN_LATENT_AVG = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
    -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
    -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
     0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667,
]

WAN_LATENT_STDDEV = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]

WAN21_LATENT_CHANNEL_COUNT = 16

WAN21_VAE_CHANNEL_DIM = 96

WAN21_VAE_SPACE_STRIDE = 8

WAN21_LATENT_AVG = [
    -0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
     0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921,
]

WAN21_LATENT_STDDEV = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]

CFG_GUIDANCE_WEIGHT = None

DEFAULT_VIDEO_AREA_CAP = None

DIFFUSION_STEP_COUNT = None

EVAL_INTERVAL = None


# ===== shared functions =====

def resolve_wan_version(args) -> str:
    """Return '2.1' (Wan2.1-1.3B) or '2.2' (Wan2.2-5B); auto-detect from the model path."""
    v = getattr(args, 'wan_version', 'auto')
    if v in ('2.1', '2.2'):
        return v
    paths = f"{getattr(args, 'pretrained_model_path', '') or ''} {getattr(args, 'vae_path', '') or ''} {getattr(args, 'wan_base', '') or ''}"
    return '2.1' if ('Wan2.1' in paths or '1.3B' in paths or '1_3B' in paths) else '2.2'

class LowRankLinear(nn.Module):
    """LoRA wrapper for nn.Linear: y = W*x + (alpha/r) * B @ (A @ x). W frozen, A kaiming, B zero-init."""
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_f, out_f = base.in_features, base.out_features
        _dtype = base.weight.dtype
        _device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, dtype=_dtype, device=_device))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, dtype=_dtype, device=_device))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._lora_enabled = True   # used by DMD — when off only goes through base (real teacher)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base(x)
        if not self._lora_enabled:        # DMD: real_score = base-only (LoRA off)
            return out
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return out + lora_out * self.scaling

def _swap_in_lora(parent: nn.Module, child_name: str, base_linear: nn.Linear, rank: int, alpha: int):
    setattr(parent, child_name, LowRankLinear(base_linear, rank, alpha))

def inject_lora_into_wan(transformer: nn.Module, args) -> nn.Module:
    """Add LoRA to the Wan transformer, base fully frozen and only train LoRA.

    Two modes:
      A. args.lora_targets in {'all','all_linear','*'}: wrap all nn.Linear.
      B. comma-separated suffix list (e.g., 'self_attn.q,...'): wrap Linears whose names end with these suffixes.
    Returns the in-place wrapped transformer.
    """
    rank, alpha = args.lora_rank, args.lora_alpha
    raw = args.lora_targets.strip()
    wrap_all_mode = raw.lower() in ('all', 'all_linear', '*')

    for p in transformer.parameters():
        p.requires_grad = False

    to_replace = []
    if wrap_all_mode:
        targets = ['*ALL_LINEAR*']
        for full_name, mod in transformer.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            parts = full_name.rsplit('.', 1)
            if len(parts) == 1:
                parent, child = transformer, parts[0]
            else:
                parent_name, child = parts
                parent = transformer.get_submodule(parent_name)
            to_replace.append((parent, child, mod, full_name, '*'))
    else:
        targets = [t.strip() for t in raw.split(',') if t.strip()]
        for full_name, mod in transformer.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            match = None
            for t in targets:
                if full_name == t or full_name.endswith('.' + t):
                    match = t
                    break
            if match is None:
                continue
            parts = full_name.rsplit('.', 1)
            if len(parts) == 1:
                parent, child = transformer, parts[0]
            else:
                parent_name, child = parts
                parent = transformer.get_submodule(parent_name)
            to_replace.append((parent, child, mod, full_name, match))

    wrapped = []
    for parent, child, base_lin, full_name, match in to_replace:
        _swap_in_lora(parent, child, base_lin, rank, alpha)
        wrapped.append((full_name, match))

    # enable patch_embedding gradients (used by the history LR branch)
    _unfrozen_extra = []
    if hasattr(transformer, 'patch_embedding'):
        for p in transformer.patch_embedding.parameters():
            p.requires_grad = True
        _unfrozen_extra.append(('patch_embedding',
                                sum(p.numel() for p in transformer.patch_embedding.parameters())))

    n_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in transformer.parameters())
    print(f"[LoRA-manual] rank={rank} alpha={alpha} scaling={alpha/rank:.3f}")
    print(f"[LoRA-manual] mode={'wrap-all-Linear' if wrap_all_mode else 'suffix-match'} targets={targets}")
    print(f"[LoRA-manual] wrapped {len(wrapped)} nn.Linear")
    if not wrap_all_mode:
        by_target = {}
        for name, m in wrapped:
            by_target[m] = by_target.get(m, 0) + 1
        for t in targets:
            if by_target.get(t, 0) == 0:
                print(f"           NO MATCH {t}")
    if _unfrozen_extra:
        for _name, _n in _unfrozen_extra:
            print(f"[LoRA-manual] extra unfreeze {_name}: {_n/1e6:.2f}M params")
    print(f"[LoRA-manual] trainable {n_trainable/1e6:.2f}M / {n_total/1e6:.1f}M total")
    return transformer

def toggle_lora_active(model: nn.Module, enabled: bool):
    """Toggle all LowRankLinear modules (DMD shares one base: off=real teacher, on=fake critic)."""
    n = 0
    for m in model.modules():
        if isinstance(m, LowRankLinear):
            m._lora_enabled = enabled
            n += 1
    return n

def register_lora_args(parser: argparse.ArgumentParser) -> None:
    """Register LoRA CLI args."""
    g = parser.add_argument_group("LoRA")
    g.add_argument("--use_lora", action="store_true",
                   help="add LoRA to the Wan transformer; base fully frozen, only train LoRA adapters")
    g.add_argument("--lora_rank", type=int, default=128)
    g.add_argument("--lora_alpha", type=int, default=128)
    g.add_argument("--lora_targets", type=str,
                   default="self_attn.q,self_attn.k,self_attn.v,self_attn.o,cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o,ffn.0,ffn.2",
                   help="LoRA target modules (Wan attention block naming)")

class WanVAEAdapter:
    """Wan2.2 VAE unified wrapper."""

    def __init__(self, model, scale, dtype, device):
        self.model = model
        self.scale = scale
        self.dtype = dtype
        self.device = device
        self.z_dim = WAN_LATENT_CHANNEL_COUNT

    def encode(self, videos):
        with torch.amp.autocast('cuda', dtype=self.dtype):
            return [
                self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                for u in videos
            ]

    def decode(self, zs):
        with torch.amp.autocast('cuda', dtype=self.dtype):
            return [
                self.model.decode(u.unsqueeze(0), self.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]

    def parameters(self):
        return self.model.parameters()

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

    def eval(self):
        self.model.eval()
        return self

def _read_sharded_safetensors(model_dir: str) -> dict:
    import glob
    state_dict = {}
    index_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index.get('weight_map', {}).values()))
        for shard_file in shard_files:
            shard_path = os.path.join(model_dir, shard_file)
            echo_on_main_rank(f"  Loading shard: {shard_file}")
            shard_dict = safetensors.torch.load_file(shard_path, device='cpu')
            state_dict.update(shard_dict)
            del shard_dict
        return state_dict
    single_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
    if os.path.exists(single_path):
        return safetensors.torch.load_file(single_path, device='cpu')
    shard_pattern = os.path.join(model_dir, "diffusion_pytorch_model-*.safetensors")
    shard_files = sorted(glob.glob(shard_pattern))
    if shard_files:
        for shard_path in shard_files:
            echo_on_main_rank(f"  Loading shard: {os.path.basename(shard_path)}")
            shard_dict = safetensors.torch.load_file(shard_path, device='cpu')
            state_dict.update(shard_dict)
            del shard_dict
        return state_dict
    raise FileNotFoundError(f"Cannot find model weights: {model_dir}")

def _resolve_vae_pth(model_path: str, fname: str) -> str:
    if os.path.isdir(model_path):
        cand = os.path.join(model_path, fname)
        if os.path.exists(cand):
            return cand
    elif os.path.isfile(model_path) and model_path.endswith('.pth'):
        return model_path
    raise FileNotFoundError(f"[Wan VAE] Cannot find {fname}: model_path={model_path}")

def build_optimizer_and_scheduler(
    model: nn.Module,
    args: argparse.Namespace,
    extra_params: list = None,
) -> Tuple[torch.optim.Optimizer, Any]:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if extra_params:
        trainable_params = trainable_params + list(extra_params)
    total_trainable = sum(p.numel() for p in trainable_params)
    echo_on_main_rank(f"--> Trainable params: {total_trainable:,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_training_steps = getattr(args, 'max_train_epochs', 100) * 100000
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, lr_scheduler

def build_train_loader(
    args: argparse.Namespace,
    skip_epochs: int = 0,
) -> Tuple[DataLoader, DistributedSampler, int]:
    echo_on_main_rank("--> Using BiwmCamCaptionData (dataset/videos, pure discrete latent camera)")
    biwm_ds = BiwmCamCaptionData(
        video_dir=args.biwm_video_dir,
        caption_json=args.biwm_caption_json,
        width=args.num_width,
        height=args.num_height,
        num_frames=args.num_frames,
        vae_temporal_factor=WAN_VAE_TIME_STRIDE,
        max_samples=args.max_train_samples,
    )
    if cp_is_active():
        _cp_sz = fetch_cp_world_size()
        _dp_size = dist.get_world_size() // _cp_sz
        _dp_rank = dist.get_rank() // _cp_sz
        sampler = CheckpointableDistSampler(biwm_ds, num_replicas=_dp_size, rank=_dp_rank, shuffle=True, drop_last=True)
    else:
        sampler = CheckpointableDistSampler(biwm_ds, shuffle=True, drop_last=True)
    if skip_epochs > 0:
        sampler.set_epoch(skip_epochs)
    _nw = args.dataloader_num_workers
    _pf = getattr(args, 'prefetch_factor', 2)
    dataloader = DataLoader(
        biwm_ds,
        sampler=sampler,
        collate_fn=biwm_collate,
        batch_size=args.train_batch_size,
        num_workers=_nw,
        drop_last=True,
        timeout=600 if _nw > 0 else 0,
        persistent_workers=_nw > 0,
        prefetch_factor=_pf if _nw > 0 else None,
    )
    return dataloader, sampler, len(biwm_ds)

def echo_on_main_rank(message: str) -> None:
    if int(os.environ.get("LOCAL_RANK", 0)) <= 0:
        print(message)

def embed_text_node_shared(
    text_encoder: nn.Module,
    encode_fn: callable,
    prompt_text: str,
    local_rank: int,
    num_local_ranks: int,
    node_group,
    node_leader_rank: int,
    device: torch.device,
    offload_te: bool = False,
) -> torch.Tensor:
    """Intra-node shared text encoding: gather all ranks' prompts, leader encodes them all, each rank fetches its own result."""
    if local_rank == 0:  # leader: collect all prompts, encode, broadcast
        all_prompts = [None] * num_local_ranks
        dist.all_gather_object(all_prompts, prompt_text, group=node_group)

        # Encode all prompts
        if offload_te and text_encoder is not None:
            te_module = getattr(text_encoder, 'model', text_encoder)
            te_module.to(device)
        with torch.no_grad():
            all_contexts = []
            for p in all_prompts:
                ctx = encode_fn([p])
                all_contexts.append(ctx[0].cpu())  # move to CPU for gloo broadcast
        if offload_te and text_encoder is not None:
            te_module = getattr(text_encoder, 'model', text_encoder)
            te_module.to("cpu")
            free_gpu_memory()

        # Broadcast metadata: [max_seq_len, hidden_dim, seq_len_0, ..., seq_len_N] (CPU tensor)
        max_seq_len = max(c.shape[0] for c in all_contexts)
        hidden_dim = all_contexts[0].shape[-1]
        meta = torch.zeros(2 + num_local_ranks, dtype=torch.long)  # CPU
        meta[0] = max_seq_len
        meta[1] = hidden_dim
        for i, c in enumerate(all_contexts):
            meta[2 + i] = c.shape[0]
        dist.broadcast(meta, src=node_leader_rank, group=node_group)

        # Pad and broadcast all encoded results (CPU tensor, gloo backend)
        padded = torch.zeros(num_local_ranks, max_seq_len, hidden_dim,
                             dtype=all_contexts[0].dtype)  # CPU
        for i, c in enumerate(all_contexts):
            padded[i, :c.shape[0]] = c
        dist.broadcast(padded, src=node_leader_rank, group=node_group)

        # Extract its own encoded result and move to GPU
        my_len = int(meta[2 + local_rank].item())
        return padded[local_rank, :my_len].to(device).contiguous()

    else:  # non-leader: participate in collective communication, receive the encoded results
        # Participate in all_gather_object (send its own prompt)
        dist.all_gather_object([None] * num_local_ranks, prompt_text, group=node_group)

        # Receive metadata (CPU tensor)
        meta = torch.zeros(2 + num_local_ranks, dtype=torch.long)  # CPU
        dist.broadcast(meta, src=node_leader_rank, group=node_group)
        max_seq_len = int(meta[0].item())
        hidden_dim = int(meta[1].item())

        # Receive encoded results (CPU tensor)
        padded = torch.zeros(num_local_ranks, max_seq_len, hidden_dim,
                             dtype=torch.bfloat16)  # CPU
        dist.broadcast(padded, src=node_leader_rank, group=node_group)

        # Extract its own encoded result and move to GPU
        my_len = int(meta[2 + local_rank].item())
        return padded[local_rank, :my_len].to(device).contiguous()

def enclose_wan_model_with_fsdp(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        CPUOffload,
        BackwardPrefetch,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import functools

    sharding_strategy = ShardingStrategy.FULL_SHARD
    echo_on_main_rank(f"--> Configuring FSDP FULL_SHARD...")

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={WanAttentionBlock},
    )
    echo_on_main_rank("--> Using transformer_auto_wrap_policy (WanAttentionBlock)")

    fsdp_kwargs = {
        "mixed_precision": mp_policy,
        "sharding_strategy": sharding_strategy,
        "device_id": device,
        "use_orig_params": True,
        "auto_wrap_policy": auto_wrap_policy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "forward_prefetch": True,
        "limit_all_gathers": True,
        "sync_module_states": True,
    }

    use_cpu_offload = getattr(args, 'use_cpu_offload', False)
    if use_cpu_offload:
        fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)
        echo_on_main_rank("--> CPU offload enabled")

    model = FSDP(model, **fsdp_kwargs)

    if getattr(args, 'gradient_checkpointing', False):
        target = model
        if hasattr(target, "module"):
            target = target.module
        elif hasattr(target, "_fsdp_wrapped_module"):
            target = target._fsdp_wrapped_module
            if hasattr(target, "module"):
                target = target.module
        if hasattr(target, '_set_gradient_checkpointing'):
            target._set_gradient_checkpointing(enable=True)
            echo_on_main_rank("--> Gradient checkpointing enabled")
        else:
            # WanModel doesn't declare _supports_gradient_checkpointing; patch each block manually.
            from torch.utils.checkpoint import checkpoint as torch_checkpoint
            gc_patched = False
            for name, module in target.named_modules():
                if hasattr(module, 'blocks') and isinstance(module.blocks, nn.ModuleList):
                    original_forward_list = []
                    for i, block in enumerate(module.blocks):
                        original_forward = block.forward
                        def make_gc_forward(orig_fn, blk):
                            def gc_forward(*args, **kwargs):
                                if blk.training:
                                    return torch_checkpoint(orig_fn, *args, use_reentrant=False, **kwargs)
                                else:
                                    return orig_fn(*args, **kwargs)
                            return gc_forward
                        block.forward = make_gc_forward(original_forward, block)
                    gc_patched = True
                    echo_on_main_rank(f"--> Gradient checkpointing enabled (manual) on {name}.blocks ({len(module.blocks)} blocks)")
                    break
            if not gc_patched:
                echo_on_main_rank("--> [Warning] Model does not support gradient checkpointing, skipping")

    return model

def free_gpu_memory() -> None:
    torch.cuda.empty_cache()
    gc.collect()

def init_wan_text_encoder(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
):
    echo_on_main_rank(f"[Wan TextEncoder] Loading: {model_path}")
    text_encoder = None
    tokenizer = None

    # Strategy 1: Wan original .pth format
    if os.path.isdir(model_path):
        t5_pth = os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth")
        tok_dir = os.path.join(model_path, "google", "umt5-xxl")
        if os.path.exists(t5_pth) and os.path.isdir(tok_dir):
            echo_on_main_rank(f"[Wan TextEncoder] Using Wan .pth format")
            from wan.modules.t5 import T5EncoderModel as WanT5EncoderModel
            text_encoder = WanT5EncoderModel(
                text_len=WAN_TEXT_LENGTH,
                dtype=dtype,
                device=device,
                checkpoint_path=t5_pth,
                tokenizer_path=tok_dir,
            )
            te_params = sum(p.numel() for p in text_encoder.model.parameters())
            echo_on_main_rank(f"[Wan TextEncoder] Params: {te_params / 1e9:.2f}B")

            def encode_fn(prompts: List[str], max_length: int = WAN_TEXT_LENGTH) -> List[torch.Tensor]:
                with torch.no_grad():
                    results = text_encoder(prompts, device)
                return [r.to(dtype=torch.bfloat16) for r in results]

            return text_encoder, text_encoder.tokenizer, encode_fn

    # Strategy 2: HF transformers format
    from transformers import AutoTokenizer, T5EncoderModel as HFT5EncoderModel
    te_path = model_path
    if os.path.isdir(model_path):
        te_subdir = os.path.join(model_path, "text_encoder")
        if os.path.isdir(te_subdir):
            te_path = te_subdir
    tokenizer_path = model_path
    if os.path.isdir(model_path):
        tok_subdir = os.path.join(model_path, "tokenizer")
        if os.path.isdir(tok_subdir):
            tokenizer_path = tok_subdir
        tok_google = os.path.join(model_path, "google", "umt5-xxl")
        if os.path.isdir(tok_google):
            tokenizer_path = tok_google
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    text_encoder = HFT5EncoderModel.from_pretrained(te_path, torch_dtype=dtype)
    text_encoder = text_encoder.to(device=device, dtype=dtype).eval()
    te_params = sum(p.numel() for p in text_encoder.parameters())
    echo_on_main_rank(f"[Wan TextEncoder] Params: {te_params / 1e9:.2f}B (HF format)")

    def encode_fn(prompts: List[str], max_length: int = WAN_TEXT_LENGTH) -> List[torch.Tensor]:
        results = []
        for prompt in prompts:
            inputs = tokenizer(
                prompt, max_length=max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            input_ids = inputs.input_ids.to(device=text_encoder.device)
            attention_mask = inputs.attention_mask.to(device=text_encoder.device)
            with torch.no_grad():
                outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                text_emb = outputs.last_hidden_state.squeeze(0)
            actual_len = attention_mask.sum().item()
            text_emb = text_emb[:int(actual_len)]
            results.append(text_emb)
        return results

    return text_encoder, tokenizer, encode_fn

def init_wan_transformer(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    model_type: str = 'ti2v',
    version: str = '2.2',
) -> WanModel:
    _tag = "Wan2.1-1.3B" if version == '2.1' else "Wan2.2-5B"
    echo_on_main_rank(f"[{_tag}] Loading transformer: {model_path}")
    config = dict(WAN_TI2V_5B_SETTINGS)
    config_path = os.path.join(model_path, "config.json") if os.path.isdir(model_path) else None
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            file_config = json.load(f)
        for k, v in file_config.items():
            if not k.startswith('_'):
                config[k] = v
        echo_on_main_rank(f"[{_tag}] Config from config.json: dim={config.get('dim')}, layers={config.get('num_layers')}")
    if model_type:
        config['model_type'] = model_type
    model = WanModel(**config)
    if os.path.isfile(model_path) and model_path.endswith('.safetensors'):
        state_dict = safetensors.torch.load_file(model_path, device='cpu')
    elif os.path.isdir(model_path):
        state_dict = _read_sharded_safetensors(model_path)
    else:
        raise FileNotFoundError(f"Cannot find model: {model_path}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        non_camera_missing = [k for k in missing if 'control_adapter' not in k and 'hycam' not in k]
        if non_camera_missing:
            echo_on_main_rank(f"[{_tag}] Missing keys ({len(non_camera_missing)}): {non_camera_missing[:5]}...")
    if unexpected:
        echo_on_main_rank(f"[{_tag}] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    del state_dict
    model = model.to(device=device, dtype=dtype)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    echo_on_main_rank(f"[{_tag}] Model params: {total_params / 1e9:.2f}B")
    return model

def init_wan_vae(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    version: str = '2.2',
):
    if version == '2.1':
        echo_on_main_rank(f"[Wan2.1-1.3B VAE] Loading Wan2.1 VAE (z_dim=16, 8x spatial): {model_path}")
        vae_pth = _resolve_vae_pth(model_path, "Wan2.1_VAE.pth")
        from wan.modules.vae2_1 import _video_vae
        vae_model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=WAN21_LATENT_CHANNEL_COUNT,
        ).eval().requires_grad_(False).to(device)
        mean_t = torch.tensor(WAN21_LATENT_AVG, dtype=dtype, device=device)
        std_t = torch.tensor(WAN21_LATENT_STDDEV, dtype=dtype, device=device)
        z_dim = WAN21_LATENT_CHANNEL_COUNT
        _tag = "Wan2.1-1.3B VAE"
    else:
        echo_on_main_rank(f"[Wan2.2-5B VAE] Loading Wan2.2 VAE (z_dim=48, 16x spatial): {model_path}")
        vae_pth = _resolve_vae_pth(model_path, "Wan2.2_VAE.pth")
        from wan.modules.vae2_2 import _video_vae
        vae_model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=WAN_LATENT_CHANNEL_COUNT,
            dim=WAN_VAE_CHANNEL_DIM,
            temperal_downsample=[False, True, True],
        ).eval().requires_grad_(False).to(device)
        mean_t = torch.tensor(WAN_LATENT_AVG, dtype=dtype, device=device)
        std_t = torch.tensor(WAN_LATENT_STDDEV, dtype=dtype, device=device)
        z_dim = WAN_LATENT_CHANNEL_COUNT
        _tag = "Wan2.2-5B VAE"
    scale = [mean_t, 1.0 / std_t]
    vae_wrapper = WanVAEAdapter(vae_model, scale, dtype, device)
    vae_wrapper.z_dim = z_dim
    vae_params = sum(p.numel() for p in vae_model.parameters())
    echo_on_main_rank(f"[{_tag}] Params: {vae_params / 1e6:.1f}M, z_dim={z_dim}")
    return vae_wrapper, vae_wrapper

def parse_step_from_checkpoint(checkpoint_path: str) -> int:
    import re
    match = re.search(r'checkpoint-(\d+)', checkpoint_path)
    if match:
        return int(match.group(1))
    return 0

def read_checkpoint_for_resume(
    checkpoint_path: str,
    model: nn.Module,
    rank: int,
) -> int:
    echo_on_main_rank(f"--> Loading checkpoint: {checkpoint_path}")
    if os.path.isdir(checkpoint_path):
        weight_path = os.path.join(checkpoint_path, "diffusion_pytorch_model.safetensors")
    else:
        weight_path = checkpoint_path
    if not os.path.exists(weight_path):
        echo_on_main_rank(f"--> [Warning] Checkpoint not found: {weight_path}")
        return 0
    state_dict = safetensors.torch.load_file(weight_path, device='cpu')
    model_keys = set(k for k, _ in model.named_parameters())
    model_keys.update(k for k, _ in model.named_buffers())
    ckpt_keys = set(state_dict.keys())
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    loaded_keys = ckpt_keys - set(unexpected)
    if rank == 0:
        print(f"[Resume] Checkpoint: {len(ckpt_keys)} keys, Model: {len(model_keys)} keys, Loaded: {len(loaded_keys)} keys")
        if missing:
            print(f"[Resume] Missing keys ({len(missing)}):")
            for k in missing:
                print(f"  MISSING: {k}")
        else:
            print(f"[Resume] All keys loaded successfully (0 missing)")
        if unexpected:
            print(f"[Resume] Unexpected keys ({len(unexpected)}):")
            for k in unexpected[:20]:
                print(f"  UNEXPECTED: {k}")
            if len(unexpected) > 20:
                print(f"  ... and {len(unexpected) - 20} more")
    del state_dict
    free_gpu_memory()
    resumed_step = parse_step_from_checkpoint(checkpoint_path)
    echo_on_main_rank(f"--> Resuming from step {resumed_step}")
    return resumed_step

def setup_wan_model(
    args: argparse.Namespace,
    local_rank: int,
    global_rank: int,
):
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16

    # ---- Resolve Wan backbone version (2.1-1.3B vs 2.2-5B) and set version-dependent latent specs ----
    version = resolve_wan_version(args)
    global WAN_LATENT_CHANNEL_COUNT, WAN_VAE_CHANNEL_DIM, WAN_VAE_SPACE_STRIDE
    if version == '2.1':
        WAN_LATENT_CHANNEL_COUNT = WAN21_LATENT_CHANNEL_COUNT   # 16
        WAN_VAE_CHANNEL_DIM = WAN21_VAE_CHANNEL_DIM             # 96
        WAN_VAE_SPACE_STRIDE = WAN21_VAE_SPACE_STRIDE           # 8
    _banner = "Wan2.1-1.3B" if version == '2.1' else "Wan2.2-5B"
    echo_on_main_rank(f"========== Backbone: {_banner} (latent z_dim={WAN_LATENT_CHANNEL_COUNT}, spatial /{WAN_VAE_SPACE_STRIDE}) ==========")

    model_type = getattr(args, 'wan_model_type', 'ti2v')
    transformer = init_wan_transformer(args.pretrained_model_path, device, dtype, model_type=model_type, version=version)

    if getattr(args, 'use_camera_control', False):
        echo_on_main_rank("[Wan] Initializing camera control...")
        transformer.add_camera_control_parameters(
            num_actions=getattr(args, 'num_actions', 81),
        )
        transformer = transformer.to(device=device, dtype=dtype)
        # --- Discrete camera config confirmation ---
        _no_discrete = getattr(args, 'no_discrete_camera', False)
        echo_on_main_rank(f"[Wan][Camera] no_discrete_camera={_no_discrete}")
        if _no_discrete:
            echo_on_main_rank("[Wan][Camera] >>> Discrete camera injection disabled (no_discrete_camera=True)")

    vae_path = getattr(args, 'vae_path', None) or args.pretrained_model_path
    vae_encoder, vae_decoder = init_wan_vae(vae_path, device, dtype, version=version)

    te_path = getattr(args, 'text_encoder_path', None) or args.pretrained_model_path
    text_encoder, tokenizer, encode_fn = init_wan_text_encoder(te_path, device, dtype)

    # Pre-encode camera action texts if cam_text mode is enabled
    if getattr(args, 'use_camera_control', False):
        echo_on_main_rank("[Wan] Pre-encoding camera action texts...")
        transformer.precompute_cam_text_embeddings(encode_fn, device, dtype)

    history_encoder = None

    # LoRA fine-tuning (optional, parameter-efficient): fine-tune low-rank adapters instead of the full DiT.
    if getattr(args, 'use_lora', False):
        echo_on_main_rank(f"[Wan] Applying LoRA r={args.lora_rank} alpha={args.lora_alpha}...")
        transformer = inject_lora_into_wan(transformer, args)

    return transformer, vae_encoder, vae_decoder, text_encoder, tokenizer, encode_fn, history_encoder

def stitch_videos_with_labels(
    left_frames: List[np.ndarray],
    right_frames: List[np.ndarray],
    output_path: str,
    left_label: str = "Ground Truth",
    right_label: str = "Generated",
    fps: int = 16,
    font_scale: float = 0.7,
    label_height: int = 40,
) -> None:
    from torchvision.io import write_video
    if len(left_frames) != len(right_frames):
        min_frames = min(len(left_frames), len(right_frames))
        left_frames = left_frames[:min_frames]
        right_frames = right_frames[:min_frames]
    import importlib.util
    has_cv2 = importlib.util.find_spec("cv2") is not None
    if has_cv2:
        import cv2
    combined_frames = []
    for left_frame, right_frame in zip(left_frames, right_frames):
        left_h, left_w = left_frame.shape[:2]
        right_h, right_w = right_frame.shape[:2]
        target_h = min(left_h, right_h)
        if target_h % 2 != 0:
            target_h -= 1
        if left_h != target_h and has_cv2:
            scale = target_h / left_h
            new_w = int(left_w * scale)
            if new_w % 2 != 0:
                new_w -= 1
            left_frame = cv2.resize(left_frame, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        if right_h != target_h and has_cv2:
            scale = target_h / right_h
            new_w = int(right_w * scale)
            if new_w % 2 != 0:
                new_w -= 1
            right_frame = cv2.resize(right_frame, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        if has_cv2:
            left_h_new, left_w_new = left_frame.shape[:2]
            right_h_new, right_w_new = right_frame.shape[:2]
            left_with_label = np.zeros((left_h_new + label_height, left_w_new, 3), dtype=np.uint8)
            left_with_label[label_height:] = left_frame
            right_with_label = np.zeros((right_h_new + label_height, right_w_new, 3), dtype=np.uint8)
            right_with_label[label_height:] = right_frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 2
            (tw, th), _ = cv2.getTextSize(left_label, font, font_scale, thickness)
            cv2.putText(left_with_label, left_label, ((left_w_new - tw) // 2, (label_height + th) // 2),
                       font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(right_label, font, font_scale, thickness)
            cv2.putText(right_with_label, right_label, ((right_w_new - tw) // 2, (label_height + th) // 2),
                       font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            combined = np.concatenate([left_with_label, right_with_label], axis=1)
        else:
            combined = np.concatenate([left_frame, right_frame], axis=1)
        combined_frames.append(combined)
    combined_tensor = torch.from_numpy(np.stack(combined_frames))
    write_video(output_path, combined_tensor, fps=int(fps), options={'crf': '18', 'preset': 'veryfast'})
    print(f"[Video] Combined video saved: {output_path}")

def store_video(
    video_frames: List[np.ndarray],
    output_path: str,
    fps: int = 16,
) -> None:
    from torchvision.io import write_video
    video_tensor = torch.from_numpy(np.stack(video_frames))
    write_video(output_path, video_tensor, fps=int(fps), options={'crf': '18', 'preset': 'veryfast'})

def wan_vae_compress(vae_encoder, video_pixels, device, dtype=torch.bfloat16):
    """Wan2.2 VAE encode: [C, T, H, W] -> [48, T_lat, H_lat, W_lat]"""
    with torch.no_grad():
        video_input = video_pixels.to(device=device, dtype=torch.float32)
        latents = vae_encoder.encode([video_input])
        video_latent = latents[0]
    return video_latent

def wan_vae_reconstruct(vae_decoder, latent, device, dtype=torch.bfloat16):
    """Wan2.2 VAE decode: [48, T_lat, H_lat, W_lat] -> List[np.ndarray]"""
    with torch.no_grad():
        # Clear VAE internal cache before each decode (encode leaves stale cache)
        _model = vae_decoder.model if hasattr(vae_decoder, 'model') else vae_decoder
        if hasattr(_model, 'clear_cache'):
            _model.clear_cache()
        latent_input = latent.to(device=device, dtype=torch.float32)
        videos = vae_decoder.decode([latent_input])
        video = videos[0]
        video = (video.clamp(-1, 1) + 1) / 2
        video = (video * 255).to(torch.uint8)
        video = video.permute(1, 2, 3, 0).cpu().numpy()
    frames = [video[i] for i in range(video.shape[0])]
    return frames

class CheckpointableDistSampler(DistributedSampler):
    """Distributed sampler that supports zero-cost fast skipping of batches."""

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.skip_batches = 0
        self.batch_size = 1

    def set_skip_batches(self, skip_batches: int, batch_size: int):
        self.skip_batches = skip_batches
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(super().__iter__())
        skip_samples = self.skip_batches * self.batch_size
        if skip_samples > 0:
            if skip_samples < len(indices):
                indices = indices[skip_samples:]
            else:
                indices = []
            self.skip_batches = 0
        return iter(indices)

def fit_frames_for_wan(total_frames: int, verbose: bool = False) -> int:
    """Adjust the frame count to satisfy the Wan VAE requirement: (frames - 1) % 4 == 0"""
    if (total_frames - 1) % WAN_VAE_TIME_STRIDE == 0:
        return total_frames
    adjusted = ((total_frames - 1) // WAN_VAE_TIME_STRIDE) * WAN_VAE_TIME_STRIDE + 1
    if adjusted < 1:
        adjusted = 1
    if verbose:
        print(f"[Wan] Frame adjustment: {total_frames} -> {adjusted}")
    return adjusted
