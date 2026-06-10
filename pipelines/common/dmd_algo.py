# vendored from minWM shared/algorithms/dmd.py
"""Distribution Matching Distillation (DMD): KL gradient, generator loss, critic loss."""

import torch
from torch import Tensor
from typing import Optional


def run_cfg(
    v_cond: Tensor,
    v_uncond: Tensor,
    guidance_scale: float,
) -> Tensor:
    """Standard Classifier-Free Guidance.

    result = uncond + scale * (cond - uncond)

    Args:
        v_cond: conditional prediction
        v_uncond: unconditional prediction
        guidance_scale: CFG scale factor
    """
    return v_uncond + guidance_scale * (v_cond - v_uncond)


def calc_kl_gradient(
    fake_x0: Tensor,
    real_x0: Tensor,
    generator_output: Tensor,
    normalize_mode: str = "global",
) -> Tensor:
    """Compute KL gradient for DMD (eq. 7 in https://arxiv.org/abs/2311.18828).

    grad = fake_x0 - real_x0, optionally normalized by |generator_output - real_x0|.

    Args:
        fake_x0: fake score's x0 prediction (after CFG if applicable)
        real_x0: real score's x0 prediction (after CFG)
        generator_output: the generator's estimated clean output (for normalization)
        normalize_mode:
            "global" — normalize by mean over all dims except batch (HY15)
            "per_sample" — normalize by mean over spatial dims, keep batch (Wan21)
            "none" — no normalization
    """
    grad = fake_x0 - real_x0

    if normalize_mode == "none":
        pass
    elif normalize_mode == "global":
        p_real = generator_output - real_x0
        normalizer = torch.abs(p_real).mean()
        grad = grad / normalizer.clamp(min=1e-8)
    elif normalize_mode == "per_sample":
        p_real = generator_output - real_x0
        # Mean over [F, C, H, W] dims, keep batch dim
        normalizer = torch.abs(p_real).mean(
            dim=list(range(1, p_real.dim())), keepdim=True
        )
        grad = grad / normalizer.clamp(min=1e-8)
    else:
        raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

    return torch.nan_to_num(grad)


def dmd_student_loss(
    generator_output: Tensor,
    kl_gradient: Tensor,
    gradient_mask: Optional[Tensor] = None,
    loss_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """DMD generator loss: 0.5 * MSE(x, x - grad).

    The gradient only flows through generator_output; the target
    (generator_output - kl_gradient) is detached.

    Args:
        generator_output: generator's clean prediction [B, F, C, H, W]
        kl_gradient: KL gradient from calc_kl_gradient (detached)
        gradient_mask: optional boolean mask for selective loss
        loss_dtype: dtype for loss computation (default float32)
    """
    x = generator_output.to(loss_dtype)
    target = (generator_output - kl_gradient).detach().to(loss_dtype)

    if gradient_mask is not None:
        x = x[gradient_mask]
        target = target[gradient_mask]

    return 0.5 * torch.nn.functional.mse_loss(x, target, reduction="mean")


def dmd_fake_score_loss(
    critic_flow_pred: Tensor,
    noise: Tensor,
    clean: Tensor,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """Critic denoising loss: MSE(flow_pred, noise - clean).

    Equivalent to flow_matching_loss(pred, noise - clean).

    Args:
        critic_flow_pred: critic's flow prediction
        noise: the noise added to the clean sample
        clean: the clean sample
        weight: optional per-element weight
    """
    target = noise - clean
    loss = (critic_flow_pred.float() - target.float()).pow(2)
    if weight is not None:
        loss = loss * weight
    return loss.mean()
