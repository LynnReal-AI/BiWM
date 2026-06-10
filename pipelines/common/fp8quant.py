# BiWM Stage 3 quantized inference: FP8-E4M3 fake-quant Linear (QAT) + FP8ScaledMMLinear (real torch._scaled_mm GEMM, per-tensor dynamic amax scaling).
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

FP8_E4M3_PEAK = 448.0
_FP8_DATATYPE = torch.float8_e4m3fn


@dataclass(slots=True)
class FP8Config:
    quantize_activations: bool = True
    skip_modules: tuple[str, ...] = field(default_factory=tuple)
    epsilon: float = 1e-6

    @classmethod
    def from_dict(cls, config) -> "FP8Config":
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        get = (config.get if isinstance(config, dict) else (lambda k, d: getattr(config, k, d)))
        return cls(
            quantize_activations=bool(get("quantize_activations", True)),
            skip_modules=tuple(get("skip_modules", ())),
            epsilon=float(get("epsilon", 1e-6)),
        )


def cast_to_fp8(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Per-tensor amax-scaled quantize to FP8-E4M3 then dequantize (same dtype)."""
    x32 = x.to(torch.float32)
    scale = torch.clamp(x32.abs().amax(), min=epsilon) / FP8_E4M3_PEAK
    q = torch.clamp(x32 / scale, -FP8_E4M3_PEAK, FP8_E4M3_PEAK).to(_FP8_DATATYPE).to(torch.float32) * scale
    return q.to(x.dtype)


def pseudo_quantize_to_fp8(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """STE fake-quant: forward=quantized, backward passes gradient straight through (QAT)."""
    q = cast_to_fp8(x, epsilon=epsilon)
    return x + (q - x).detach()


class FP8Linear(nn.Linear):
    """fake-quant FP8 Linear (QAT). When quantization_enabled=False, falls back to ordinary bf16 linear."""

    def __init__(self, linear: nn.Linear, config: FP8Config):
        super().__init__(linear.in_features, linear.out_features, bias=linear.bias is not None)
        self.quant_config = config
        self.quantization_enabled = True
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone())

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: FP8Config) -> "FP8Linear":
        return cls(linear=linear, config=config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.quantization_enabled:
            return F.linear(inputs, self.weight, self.bias)
        w = pseudo_quantize_to_fp8(self.weight, self.quant_config.epsilon)
        if self.quant_config.quantize_activations:
            inputs = pseudo_quantize_to_fp8(inputs, self.quant_config.epsilon)
        return F.linear(inputs, w, self.bias)


def _skip_layer(module_path: str, skip_modules: Iterable[str]) -> bool:
    return any(s and s in module_path for s in skip_modules)


def enable_fp8_quantization(module: nn.Module, config: FP8Config, prefix: str = "") -> list[str]:
    """Recursively replace nn.Linear with FP8Linear (fake-quant, used for QAT). Returns the replaced module paths."""
    replaced: list[str] = []
    for name, child in list(module.named_children()):
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, FP8Linear):
            continue
        if isinstance(child, nn.Linear) and not _skip_layer(path, config.skip_modules):
            setattr(module, name, FP8Linear.from_linear(child, config))
            replaced.append(path)
            continue
        replaced.extend(enable_fp8_quantization(child, config, prefix=path))
    return replaced


def toggle_fp8_quantization(module: nn.Module, enabled: bool) -> int:
    n = 0
    for child in module.modules():
        if isinstance(child, FP8Linear):
            child.quantization_enabled = enabled
            n += 1
    return n


# Real FP8 GEMM inference (Hopper/H200/Ada): torch._scaled_mm, per-tensor dynamic scale
def fp8_scaled_mm_present() -> bool:
    return hasattr(torch, "_scaled_mm") and torch.cuda.is_available()


class FP8ScaledMMLinear(nn.Module):
    """Real FP8 inference: weights quantized offline to FP8 (per-tensor), activations
    dynamically quantized in forward, via torch._scaled_mm; falls back to bf16 F.linear
    when the kernel is unavailable (dims not 16-aligned)."""

    def __init__(self, in_features, out_features, weight_fp8, weight_scale, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_fp8", weight_fp8)          # [out, in] float8_e4m3fn
        self.register_buffer("weight_scale", weight_scale)      # scalar float32
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        # compile-friendly: decide kernel usability at init (dims need 16-alignment)
        self._use_kernel = bool(fp8_scaled_mm_present()
                                and in_features % 16 == 0 and out_features % 16 == 0)

    @classmethod
    def from_linear(cls, linear: nn.Linear, epsilon: float = 1e-6) -> "FP8ScaledMMLinear":
        w = linear.weight.detach().to(torch.float32)
        wscale = torch.clamp(w.abs().amax(), min=epsilon) / FP8_E4M3_PEAK
        w_fp8 = torch.clamp(w / wscale, -FP8_E4M3_PEAK, FP8_E4M3_PEAK).to(_FP8_DATATYPE)
        bias = (linear.bias.detach().to(torch.bfloat16) if linear.bias is not None else None)
        return cls(linear.in_features, linear.out_features, w_fp8,
                   wscale.to(torch.float32), bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        x2 = x.reshape(-1, orig[-1])
        out_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        if self._use_kernel:
            # dynamic per-tensor activation quantization -> FP8xFP8 GEMM
            xscale = torch.clamp(x2.abs().amax().float(), min=1e-6) / FP8_E4M3_PEAK
            x_fp8 = torch.clamp(x2.float() / xscale, -FP8_E4M3_PEAK, FP8_E4M3_PEAK).to(_FP8_DATATYPE)
            # mat2 needs column-major: weight_fp8 [out,in] -> .t() = [in,out] col-major
            out = torch._scaled_mm(
                x_fp8, self.weight_fp8.t(),
                scale_a=xscale, scale_b=self.weight_scale,
                bias=self.bias, out_dtype=out_dtype,
            )
        else:
            # kernel unavailable: dequantize weights to bf16 ordinary GEMM
            w = self.weight_fp8.to(out_dtype) * self.weight_scale.to(out_dtype)
            out = F.linear(x2.to(out_dtype), w, self.bias)
        return out.reshape(*orig[:-1], self.out_features)


def enable_fp8_kernel_quantization(module: nn.Module, skip_modules=(), epsilon: float = 1e-6,
                                  prefix: str = "") -> list[str]:
    """Recursively replace nn.Linear with FP8ScaledMMLinear (real FP8 GEMM inference). Returns the replaced paths."""
    replaced: list[str] = []
    for name, child in list(module.named_children()):
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, FP8ScaledMMLinear):
            continue
        if isinstance(child, nn.Linear) and not _skip_layer(path, skip_modules):
            setattr(module, name, FP8ScaledMMLinear.from_linear(child, epsilon=epsilon))
            replaced.append(path)
            continue
        replaced.extend(enable_fp8_kernel_quantization(child, skip_modules, epsilon, path))
    return replaced
