# Sequence-parallel collective communication (autograd-aware), over biWM's nccl_info SP group.
import torch
import torch.distributed as dist

from pipelines.utils.parallel_states import nccl_info


def _sp_size():
    return nccl_info.sp_size


def _sp_group():
    return nccl_info.group


def _all_to_all(x, scatter_dim, gather_dim, group):
    """Split along scatter_dim into world shards, all_to_all, then concat back along gather_dim. Supports arbitrary dims (including 4D)."""
    world = dist.get_world_size(group)
    if world == 1:
        return x
    inputs = [t.contiguous() for t in x.chunk(world, dim=scatter_dim)]
    outputs = [torch.empty_like(t) for t in inputs]
    dist.all_to_all(outputs, inputs, group=group)
    return torch.cat(outputs, dim=gather_dim).contiguous()


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        world = _sp_size()
        ctx.dim = dim
        ctx.world = world
        ctx.rank = nccl_info.rank_within_group
        if world == 1:
            return x
        x = x.contiguous()
        outs = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(outs, x, group=_sp_group())
        return torch.cat(outs, dim=dim)

    @staticmethod
    def backward(ctx, grad):
        if ctx.world == 1:
            return grad, None
        # split the full-sequence gradient back along dim, this rank only takes its own share
        return grad.chunk(ctx.world, dim=ctx.dim)[ctx.rank].contiguous(), None


class _AllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scatter_dim, gather_dim):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        if _sp_size() == 1:
            return x
        return _all_to_all(x, scatter_dim, gather_dim, _sp_group())

    @staticmethod
    def backward(ctx, grad):
        if _sp_size() == 1:
            return grad, None, None
        # inverse: swap the scatter/gather dims to do the reverse all_to_all
        return (_all_to_all(grad, ctx.gather_dim, ctx.scatter_dim, _sp_group()),
                None, None)


def sequence_model_parallel_all_gather(x, dim=0):
    if _sp_size() == 1:
        return x
    return _AllGather.apply(x, dim)


def sequence_model_parallel_all_to_all_4D(x, scatter_dim=2, gather_dim=1):
    if _sp_size() == 1:
        return x
    return _AllToAll4D.apply(x, scatter_dim, gather_dim)
