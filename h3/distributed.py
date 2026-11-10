"""Gradient norm for fully sharded mixed BF16/FP32 original parameters."""
import torch
import torch.distributed as dist


@torch.no_grad()
def clip_sharded_grad_norm(parameters, max_norm=1.):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        raise RuntimeError("No gradients for this update")
    squared = torch.zeros((), device=grads[0].device, dtype=torch.float64)
    for grad in grads:
        for chunk in grad.detach().reshape(-1).split(8 << 20):
            squared += chunk.double().square().sum()
    if dist.is_initialized():
        dist.all_reduce(squared, op=dist.ReduceOp.SUM)
    norm = squared.sqrt()
    if torch.isfinite(norm):
        coefficient = (max_norm / (norm + 1e-6)).clamp(max=1.)
        for grad in grads:
            work = grad if grad.is_contiguous() else grad.contiguous()
            for chunk in work.view(-1).split(8 << 20):
                chunk.copy_(chunk.double() * coefficient)
            if work is not grad:
                grad.copy_(work)
    return norm
