"""Preserve FA3's native autograd through Diffusers' local FA3 dispatcher.

Diffusers 0.40 wraps this kernel in a forward-only custom op. The native
flash-attn-3 interface supplies the matching backward; no backend is changed.
Adapted from the SolarWM stage0.5_v7 training entry's fa3_autograd_policy.
"""
from __future__ import annotations

import torch


def install_fa3_autograd():
    from diffusers.models import attention_dispatch as dispatch

    current = dispatch._wrapped_flash_attn_3
    if getattr(current, "biwm_native_fa3_autograd", False):
        return current
    if not callable(dispatch.flash_attn_3_func):
        raise RuntimeError("Native flash-attn-3 is required; no fallback is allowed")

    def differentiable(q, k, v, softmax_scale=None, causal=False, qv=None,
                       q_descale=None, k_descale=None, v_descale=None,
                       attention_chunk=0, softcap=0.0, num_splits=1,
                       pack_gqa=None, deterministic=False, sm_margin=0):
        output, lse, *_ = dispatch.flash_attn_3_func(
            q=q, k=k, v=v, softmax_scale=softmax_scale, causal=causal, qv=qv,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            window_size=(-1, -1), attention_chunk=attention_chunk,
            softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa,
            deterministic=deterministic, sm_margin=sm_margin,
            return_attn_probs=True)
        return output, lse.permute(0, 2, 1)

    differentiable.biwm_native_fa3_autograd = True
    differentiable.original_custom_op = current
    dispatch._wrapped_flash_attn_3 = differentiable
    return differentiable


def verify_fa3_backward(device):
    """Fail before loading H3 if the selected FA3 path cannot differentiate.

    Compare the original custom-op forward exactly, then check Q/K/V gradients
    against FP32 attention at an irregular length and native H3 head size.
    The FP32 reference is only a tiny numerical gate, never a training backend.
    Preserve RNG states so this gate does not change the experiment seed.
    """
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    wrapper = install_fa3_autograd()
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(731)
        values = [torch.randn(1, 17, 2, 128, device=device,
                              dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        output = dispatch_attention_fn(*values, backend="_flash_3")
        with torch.no_grad():
            original, _ = wrapper.original_custom_op(*values)
        torch.testing.assert_close(output, original, rtol=0, atol=0)
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        reference_values = [value.detach().float().requires_grad_() for value in values]
        query, key, value = [item.transpose(1, 2) for item in reference_values]
        scores = query @ key.transpose(-1, -2) / query.shape[-1] ** .5
        reference = (scores.softmax(-1) @ value).transpose(1, 2)
        reference.backward(grad_output.float())
        errors = []
        for actual, expected in zip(values, reference_values):
            if actual.grad is None or not torch.isfinite(actual.grad).all():
                raise RuntimeError("Native FA3 Q/K/V backward produced missing/nonfinite gradients")
            error = (actual.grad.float() - expected.grad).norm() / expected.grad.norm().clamp_min(1e-12)
            errors.append(float(error))
            if error > .03:
                raise RuntimeError(f"Native FA3 gradient differs from FP32 reference: {float(error)}")
        torch.cuda.synchronize(device)
    return dict(backend="_flash_3", fallback=False, forward_max_abs_difference=0,
                qkv_gradient_relative_l2=errors, qkv_gradient_finite=True)
