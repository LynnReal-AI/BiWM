# -*- coding: utf-8 -*-
"""
Ulysses-style Context Parallel for long-video training.

Partition the input along the temporal (sequence) dimension across GPUs and use
all-to-all to redistribute activations during attention; each device locally
computes attention over its sequence shard.
"""

import datetime
import torch
import torch.distributed as dist
from typing import Optional, Tuple
from dataclasses import dataclass

# Long video training requires a longer timeout (2 hours)
_CP_DEADLINE = datetime.timedelta(hours=2)


@dataclass
class ContextParallelSettings:
    """Context Parallel configuration"""
    enabled: bool = False
    cp_size: int = 1  # Number of GPUs for Context Parallel
    cp_rank: int = 0  # Rank of the current GPU within the CP group
    cp_group: Optional[dist.ProcessGroup] = None


# Global CP configuration
_cp_settings = ContextParallelSettings()


def setup_context_parallel(cp_size: int = 1):
    """Initialize the Context Parallel group. If cp_size <= 1, CP is disabled."""
    global _cp_settings

    if cp_size <= 1:
        _cp_settings = ContextParallelSettings(enabled=False, cp_size=1, cp_rank=0, cp_group=None)
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size % cp_size == 0, f"world_size ({world_size}) must be divisible by cp_size ({cp_size})"

    num_cp_groups = world_size // cp_size
    for i in range(num_cp_groups):
        ranks = list(range(i * cp_size, (i + 1) * cp_size))
        group = dist.new_group(ranks, timeout=_CP_DEADLINE)
        if rank in ranks:
            _cp_settings = ContextParallelSettings(
                enabled=True,
                cp_size=cp_size,
                cp_rank=rank - i * cp_size,
                cp_group=group,
            )


def prime_context_parallel(hidden_dim: int = 4096, num_heads: int = 32,
                            seq_len: int = 1024, device: torch.device = None):
    """
    NCCL communication warmup.

    Perform a dummy all-to-all before the first forward, forcing NCCL to complete
    communicator initialization and ring/tree topology setup, avoiding NCCL lazy
    init causing some ranks to time out on the first forward step.
    """
    if not _cp_settings.enabled:
        return

    if device is None:
        device = torch.cuda.current_device()

    head_dim = hidden_dim // num_heads
    cp_size = _cp_settings.cp_size
    group = _cp_settings.cp_group

    # Pattern 1: scatter heads, gather sequence (before attention)
    dummy = torch.zeros(1, seq_len // cp_size, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    _ = _alltoall_single_op(dummy, scatter_dim=2, gather_dim=1, group=group)

    # Pattern 2: scatter sequence, gather heads (after attention)
    dummy2 = torch.zeros(1, seq_len, num_heads // cp_size, head_dim, device=device, dtype=torch.bfloat16)
    _ = _alltoall_single_op(dummy2, scatter_dim=1, gather_dim=2, group=group)

    # Pattern 3: all-gather (for gradient sync)
    dummy3 = torch.zeros(seq_len // cp_size, device=device, dtype=torch.bfloat16)
    gathered = [torch.zeros_like(dummy3) for _ in range(cp_size)]
    dist.all_gather(gathered, dummy3, group=group)

    dist.barrier()

    del dummy, dummy2, dummy3, gathered
    torch.cuda.empty_cache()


def fetch_cp_settings() -> ContextParallelSettings:
    """Get the Context Parallel configuration"""
    return _cp_settings


def teardown_context_parallel():
    """Destroy the Context Parallel group"""
    global _cp_settings
    _cp_settings = ContextParallelSettings()


def _alltoall_single_op(
    input_: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """All-to-all: scatter along scatter_dim, gather along gather_dim (Ulysses CP)."""
    world_size = dist.get_world_size(group)

    if world_size == 1:
        return input_

    # chunk returns views, must be contiguous for NCCL
    input_list = [chunk.contiguous() for chunk in torch.chunk(input_, world_size, dim=scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim)


def disperse_sequence(input_: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Scatter the sequence across all CP GPUs: [B,S,...] -> [B,S//cp_size,...]."""
    if not _cp_settings.enabled:
        return input_

    seq_len = input_.shape[dim]
    assert seq_len % _cp_settings.cp_size == 0,\
        f"Sequence length ({seq_len}) must be divisible by cp_size ({_cp_settings.cp_size})"

    chunks = torch.chunk(input_, _cp_settings.cp_size, dim=dim)
    return chunks[_cp_settings.cp_rank].contiguous()


def collect_sequence(input_: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Gather the sequence from all CP GPUs into the full sequence."""
    if not _cp_settings.enabled:
        return input_

    world_size = _cp_settings.cp_size
    gathered = [torch.empty_like(input_) for _ in range(world_size)]
    dist.all_gather(gathered, input_.contiguous(), group=_cp_settings.cp_group)

    return torch.cat(gathered, dim=dim)


class _GatherForwardOp(torch.autograd.Function):
    """Gather in forward, scatter in backward (for loss computation, etc.)"""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int) -> torch.Tensor:
        ctx.dim = dim
        return collect_sequence(input_, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return disperse_sequence(grad_output, ctx.dim), None


def collect_for_loss(input_: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Gather in forward for loss computation (gradients auto-scattered in backward)"""
    if not _cp_settings.enabled:
        return input_
    return _GatherForwardOp.apply(input_, dim)


def pad_for_cp_divisible(tensor: torch.Tensor, dim: int = 1) -> Tuple[torch.Tensor, int]:
    """Pad the tensor along dim to be divisible by cp_size. Returns (padded, original_length)."""
    if not _cp_settings.enabled:
        return tensor, tensor.shape[dim]

    original_length = tensor.shape[dim]
    cp_size = _cp_settings.cp_size

    if original_length % cp_size == 0:
        return tensor, original_length

    pad_length = cp_size - (original_length % cp_size)

    # PyTorch pad starts from the last dimension
    pad_spec = [0] * (2 * tensor.ndim)
    pad_idx = 2 * (tensor.ndim - 1 - dim)
    pad_spec[pad_idx + 1] = pad_length  # pad at the end of this dimension

    padded = torch.nn.functional.pad(tensor, pad_spec, mode='constant', value=0)
    return padded, original_length


def strip_cp_padding(tensor: torch.Tensor, original_length: int, dim: int = 1) -> torch.Tensor:
    """Remove CP padding, slicing back to original_length along dim."""
    if not _cp_settings.enabled or tensor.shape[dim] == original_length:
        return tensor

    return tensor.narrow(dim, 0, original_length)


def fetch_cp_world_size() -> int:
    """Get the CP world size"""
    return _cp_settings.cp_size if _cp_settings.enabled else 1


def fetch_cp_rank() -> int:
    """Get the CP rank"""
    return _cp_settings.cp_rank if _cp_settings.enabled else 0


def cp_is_active() -> bool:
    """Check whether CP is enabled"""
    return _cp_settings.enabled


def apply_ulysses_attention(q, k, v, heads, attention_fn, mask=None):
    """Run self-attention with Ulysses sequence/head all-to-all exchange."""
    if not _cp_settings.enabled:
        return attention_fn(q, k, v, heads, mask)

    batch, local_sequence, hidden = q.shape
    head_dim = hidden // heads
    assert heads % _cp_settings.cp_size == 0, (
        f"Number of heads ({heads}) must be divisible by cp_size ({_cp_settings.cp_size})"
    )

    def exchange(tensor, scatter_dim, gather_dim):
        return _alltoall_single_op(
            tensor,
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
            group=_cp_settings.cp_group,
        )

    q = exchange(q.view(batch, local_sequence, heads, head_dim), 2, 1)
    k = exchange(k.view(batch, local_sequence, heads, head_dim), 2, 1)
    v = exchange(v.view(batch, local_sequence, heads, head_dim), 2, 1)
    _, full_sequence, local_heads, _ = q.shape
    q = q.reshape(batch, full_sequence, local_heads * head_dim)
    k = k.reshape(batch, full_sequence, local_heads * head_dim)
    v = v.reshape(batch, full_sequence, local_heads * head_dim)
    if mask is not None and mask.shape[-1] == local_sequence:
        mask = collect_sequence(mask, dim=-1)
    output = attention_fn(q, k, v, local_heads, mask)
    output = output.view(batch, full_sequence, local_heads, head_dim)
    output = exchange(output, 1, 2)
    return output.reshape(batch, local_sequence, hidden)


def calc_cp_divisible_frames(frames: int, cp_size: int, temporal_stride: int = 8) -> int:
    """
    Compute a frame count divisible by both CP and temporal_stride.

    LTX2 VAE requires (frames - 1) % temporal_stride == 0; CP requires
    latent_frames % cp_size == 0.
    """
    frames = ((frames - 1) // temporal_stride) * temporal_stride + 1

    latent_frames = (frames - 1) // temporal_stride + 1
    if latent_frames % cp_size != 0:
        latent_frames = ((latent_frames // cp_size) + 1) * cp_size
        frames = (latent_frames - 1) * temporal_stride + 1

    return frames
