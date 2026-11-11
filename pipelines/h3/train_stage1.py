"""Minimal H3 camera flow-matching trainer; run with torchrun.

Full or LoRA training, FSDP sharding, FP32 master AdamW, and same-topology
checkpoint resume. Preencoding keeps Qwen/VAE out of the training process.
"""
import argparse
from datetime import timedelta
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, ShardingStrategy,
    ShardedStateDictConfig, StateDictType, MixedPrecision,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.utils.data import DataLoader

from h3.data import CameraDataset
from h3.distributed import clip_sharded_grad_norm
from h3.flow import flow_matching_loss
from h3.h3_fa3_policy import require_h3_fa3_training_backend
from h3.fa3_autograd import verify_fa3_backward
from h3.model import BiWMH3Transformer
from h3.optimizer import FP32MasterAdamW


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=("full", "lora"), default="lora")
    parser.add_argument("--lora_rank", type=int, default=384)
    parser.add_argument("--lora_alpha", type=int, default=384)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention_backend", choices=("_flash_3",), default="_flash_3")
    parser.add_argument("--resume", help="Complete checkpoint directory, same world size")
    return parser.parse_args()


def configure_lora(model, args):
    from peft import LoraConfig
    suffixes = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and name.endswith(suffixes)]
    if len(targets) != 312:
        raise RuntimeError(f"Expected 312 H3 QKVO/FFN linears, got {len(targets)}")
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                                target_modules=targets, lora_dropout=0., bias="none"))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    return targets


def wrap_fsdp(model, device):
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3TransformerBlock, MiniMaxH3TokenRefinerBlock,
    )
    # Separate FP32 projections/time MLP from BF16 blocks before flattening.
    fp32_units = {model.proj_in, model.proj_out, model.audio_proj_in,
                  model.audio_proj_out, model.time_embedder}
    policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda module:
                     module in fp32_units or isinstance(module, (
                         MiniMaxH3TransformerBlock, MiniMaxH3TokenRefinerBlock)))
    return FSDP(model, auto_wrap_policy=policy, sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=device, use_orig_params=True, sync_module_states=device.type == "cuda",
                limit_all_gathers=True,
                mixed_precision=MixedPrecision(param_dtype=None, reduce_dtype=torch.float32))


def checkpoint_context(model):
    return FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                               ShardedStateDictConfig(offload_to_cpu=True))


def save_checkpoint(output, model, optimizer, step, config, rank, world, device):
    destination = output / f"checkpoint_{step:06d}"
    destination.mkdir(exist_ok=True)
    with checkpoint_context(model):
        model_state = model.state_dict()
    state = dict(model=model_state, optimizer=optimizer.state_dict(), step=step,
                 config=config, world_size=world, torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                 python_rng=random.getstate(),
                 numpy_rng=np.random.get_state())
    temporary = destination / f"rank{rank:03d}.tmp"
    torch.save(state, temporary)
    temporary.replace(destination / f"rank{rank:03d}.pt")
    del model_state, state
    dist.barrier()
    if rank == 0:
        (destination / "COMPLETE.json").write_text(json.dumps(
            dict(step=step, world_size=world, source_weight="Live"), indent=2) + "\n")
    dist.barrier()


def resume_checkpoint(path, model, optimizer, config, rank, world, device):
    path = Path(path)
    receipt = json.loads((path / "COMPLETE.json").read_text())
    if receipt["world_size"] != world:
        raise ValueError("H3 resume requires the saved FSDP world size")
    state = torch.load(path / f"rank{rank:03d}.pt", map_location="cpu", weights_only=False)
    # Max steps and output location may change; the scientific/data contract may not.
    mutable = {"max_steps", "output_dir", "resume", "checkpointing_steps"}
    old = {k: v for k, v in state["config"].items() if k not in mutable}
    new = {k: v for k, v in config.items() if k not in mutable}
    if old != new:
        raise ValueError("Checkpoint training/data configuration differs")
    with checkpoint_context(model):
        model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    return int(state["step"])


def main():
    args = parse_args()
    if min(args.max_steps, args.gradient_accumulation_steps, args.checkpointing_steps,
           args.lora_rank, args.lora_alpha) < 1 or args.learning_rate <= 0:
        raise ValueError("Steps, LoRA dimensions, accumulation and LR must be positive")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    require_h3_fa3_training_backend(args.attention_backend, device)
    print(json.dumps(dict(event="fa3_backward_gate", **verify_fa3_backward(device))), flush=True)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    dataset = CameraDataset(args.manifest)
    if len(dataset) < world:
        raise ValueError("At least one unique data sample per rank is required")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), stage="biwm_h3_stage1_camera_fm", shift=3.0,
                  anchor_clean_time=.999, loss="video_flow_matching_MSE_only",
                  hand_condition=False, auxiliary_losses=False, audio_tokens=0,
                  framepack=False, source_weight="Live", qwen_skip_reference_videos=True,
                  world_size=world, sequence_parallel_size=1, micro_batch_size=1,
                  effective_batch_size=world * args.gradient_accumulation_steps,
                  dtype="BF16 blocks / FP32 input-output-time / FP32 master AdamW",
                  optimizer=dict(betas=[.9, .95], eps=1e-8, weight_decay=.01,
                                 gradient_clip=1., warmup=0, schedule="constant"),
                  validation="external fixed-case Live sampling, CFG1",
                  manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest())
    if not args.resume and any(output.iterdir()):
        raise ValueError("Use an empty output directory for a new run, or --resume")
    dist.barrier()
    model = BiWMH3Transformer.strict_from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True)
    model.set_attention_backend(args.attention_backend)
    if args.mode == "lora":
        config["lora_targets"] = configure_lora(model, args)
    model.enable_gradient_checkpointing()
    model = wrap_fsdp(model, device)
    model.train()
    optimizer = FP32MasterAdamW((p for p in model.parameters() if p.requires_grad),
                               lr=args.learning_rate, betas=(.9, .95), eps=1e-8, weight_decay=.01)
    step = resume_checkpoint(args.resume, model, optimizer, config, rank, world, device) if args.resume else 0
    if rank == 0:
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "step_receipts").mkdir(exist_ok=True)
    # Deterministic, disjoint rank samples with no padding, skipping or reader RNG.
    # The cursor follows the saved complete optimizer step exactly on resume.
    accumulation = args.gradient_accumulation_steps
    def sample_indices():
        cursor = step * accumulation * world + rank
        stop = args.max_steps * accumulation * world
        while cursor < stop:
            yield cursor % len(dataset)
            cursor += world
    loader = iter(DataLoader(dataset, batch_size=None, sampler=sample_indices(), num_workers=0,
                           generator=torch.Generator().manual_seed(args.seed)))
    try:
        while step < args.max_steps:
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            # Synchronize each microbatch to keep FSDP gradients sharded.
            for _ in range(accumulation):
                sample = {k: v.to(device) for k, v in next(loader).items()}
                loss = flow_matching_loss(model, sample)
                finite = torch.isfinite(loss.detach()).int()
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise RuntimeError("Nonfinite flow-matching loss; no optimizer update")
                (loss / accumulation).backward()
                loss_sum += loss.detach() / accumulation
            norm = clip_sharded_grad_norm(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite global gradient norm; no optimizer update")
            optimizer.step()
            step += 1
            dist.all_reduce(loss_sum)
            record = dict(step=step, rank=rank, loss=float(loss_sum / world), gradient_norm=float(norm))
            (output / "step_receipts" / f"COMPLETE.step{step:06d}.rank{rank:03d}.json").write_text(
                json.dumps(record) + "\n")
            if rank == 0:
                print(json.dumps(record), flush=True)
                with (output / "log.txt").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
            if step % args.checkpointing_steps == 0 or step == args.max_steps:
                save_checkpoint(output, model, optimizer, step, config, rank, world, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
