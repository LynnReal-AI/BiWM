# >>> BiWM Stage 3 — quantized inference: NVFP4 real FP4 kernel <<<
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

log_handle = logging.getLogger(__name__)

# Soft-import lightx2v_kernel (real NVFP4 CUTLASS kernels, Blackwell-only)
_KERNEL_PRESENT = False
_scaled_nvfp4_quant = None
_cutlass_scaled_nvfp4_mm = None

try:
    from lightx2v_kernel.gemm import (
        cutlass_scaled_nvfp4_mm as _cutlass_scaled_nvfp4_mm,
        scaled_nvfp4_quant as _scaled_nvfp4_quant,
    )

    _KERNEL_PRESENT = True
    log_handle.info("lightx2v_kernel NVFP4 kernels loaded successfully.")
except ImportError:
    log_handle.warning(
        "lightx2v_kernel not installed — NVFP4 kernel acceleration unavailable. "
        "Install lightx2v_kernel (requires Blackwell GPU) for real FP4 acceleration."
    )


def nvfp4_kernel_present() -> bool:
    """Return True if real NVFP4 CUTLASS kernels can be used."""
    return _KERNEL_PRESENT


SCALE_FACTOR = 2688.0


class NVFP4KernelLinear(nn.Module):
    """nn.Linear replacement using CUTLASS NVFP4 GEMM.

    Weight stored as packed uint8 FP4 + block-wise float8_e4m3fn swizzled scales
    (LightX2V scaled_nvfp4_quant format). Forward dynamically quantizes activations
    to FP4 and calls cutlass_scaled_nvfp4_mm for FP4xFP4 -> BF16 GEMM.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Packed FP4 weight: [out_features, in_features // 2]
        self.register_buffer(
            "weight",
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
        )
        # Block-wise scale in swizzled layout
        self.register_buffer(
            "weight_scale",
            torch.zeros(1, dtype=torch.float8_e4m3fn),
        )
        # Input global scale for dynamic activation quantization
        self.register_buffer(
            "input_global_scale",
            torch.tensor(1.0, dtype=torch.float32),
        )
        # Pre-computed alpha = 1 / (input_global_scale * weight_global_scale)
        self.register_buffer(
            "alpha",
            torch.tensor(1.0, dtype=torch.float32),
        )

        if bias:
            self.register_buffer(
                "bias", torch.zeros(out_features, dtype=torch.bfloat16)
            )
        else:
            self.bias = None

    def _apply(self, fn, recurse=True):
        """Prevent .to(dtype=...) from corrupting packed FP4 buffers."""
        def _safe_fn(tensor):
            result = fn(tensor)
            if result is not None and tensor.dtype in (torch.uint8, torch.float8_e4m3fn):
                if result.dtype != tensor.dtype:
                    return tensor.to(device=result.device)
            return result
        return super()._apply(_safe_fn, recurse=recurse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _KERNEL_PRESENT:
            raise RuntimeError(
                "NVFP4KernelLinear.forward() called but lightx2v_kernel is not installed. "
                "Install lightx2v_kernel on a Blackwell GPU to use real FP4 acceleration."
            )

        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        # Dynamic FP4 activation quantization
        input_quant, input_scale = _scaled_nvfp4_quant(x, self.input_global_scale)

        # CUTLASS FP4 × FP4 GEMM → BF16
        output = _cutlass_scaled_nvfp4_mm(
            input_quant,
            self.weight,
            input_scale,
            self.weight_scale,
            alpha=self.alpha,
            bias=self.bias,
        )

        if len(orig_shape) > 2:
            output = output.reshape(*orig_shape[:-1], self.out_features)

        return output

    @classmethod
    def from_linear_and_scales(
        cls,
        in_features: int,
        out_features: int,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_global_scale: torch.Tensor,
        alpha: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> "NVFP4KernelLinear":
        """Construct from pre-quantized tensors."""
        module = cls(in_features, out_features, bias=bias is not None)
        module.weight = packed_weight
        module.weight_scale = weight_scale
        module.input_global_scale = input_global_scale
        module.alpha = alpha
        if bias is not None:
            module.bias = bias
        return module

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, kernel={'available' if _KERNEL_PRESENT else 'unavailable'}"
        )


def _module_excluded(module_path: str, skip_modules: Iterable[str]) -> bool:
    return any(skip_name and skip_name in module_path for skip_name in skip_modules)


def read_nvfp4_kernel_checkpoint(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    skip_modules: tuple[str, ...] = (),
) -> list[str]:
    """Replace nn.Linear with NVFP4KernelLinear from a LightX2V-compatible NVFP4 checkpoint.

    Each quantized layer needs {prefix}.{weight (uint8 packed FP4), weight_scale
    (float8_e4m3fn swizzled), input_global_scale (f32), alpha (f32), bias (optional)}.
    Layers absent from the checkpoint or matching skip_modules are left as-is.
    Returns list of replaced module paths.
    """
    # Discover which layers have packed NVFP4 weights
    nvfp4_prefixes: set[str] = set()
    for key in state_dict:
        if key.endswith(".input_global_scale"):
            prefix = key.removesuffix(".input_global_scale")
            nvfp4_prefixes.add(prefix)

    replaced: list[str] = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, NVFP4KernelLinear):
            continue
        if _module_excluded(name, skip_modules):
            continue

        # Map Self-Forcing key names to checkpoint key names
        weight_key = f"{name}.weight"
        if name not in nvfp4_prefixes and f"model.{name}" in nvfp4_prefixes:
            prefix = f"model.{name}"
        elif name in nvfp4_prefixes:
            prefix = name
        else:
            continue

        packed_weight = state_dict[f"{prefix}.weight"]
        weight_scale = state_dict[f"{prefix}.weight_scale"]
        input_global_scale = state_dict[f"{prefix}.input_global_scale"]
        alpha = state_dict[f"{prefix}.alpha"]
        bias = state_dict.get(f"{prefix}.bias", None)

        kernel_linear = NVFP4KernelLinear.from_linear_and_scales(
            in_features=module.in_features,
            out_features=module.out_features,
            packed_weight=packed_weight,
            weight_scale=weight_scale,
            input_global_scale=input_global_scale,
            alpha=alpha,
            bias=bias,
        )

        # Replace in parent
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = model.get_submodule(parts[0])
            setattr(parent, parts[1], kernel_linear)
        else:
            setattr(model, name, kernel_linear)

        replaced.append(name)

    log_handle.info("Replaced %d layers with NVFP4KernelLinear", len(replaced))
    return replaced
