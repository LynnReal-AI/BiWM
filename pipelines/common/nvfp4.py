# BiWM Stage 3 quantized inference: NVFP4 fake-quant (STE, QAT) Linear wrapper + model-wide replacement + toggle.
# block_size=16, FP4 levels + FP8-E4M3 block scale (NVIDIA NVFP4 two-level scale). Real acceleration needs Blackwell + FP4 kernel.
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


FP4_QUANT_LEVELS = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
FP8_E4M3_PEAK = 448.0
FP4_E2M1_PEAK = 6.0   # FP4(E2M1) max representable level; block scale must normalize against it to use all 16 levels


@dataclass(slots=True)
class NVFP4Config:
    block_size: int = 16
    quantize_activations: bool = False   # default weight-only (W4A16): FP4 activation outliers hurt accuracy
    skip_modules: tuple[str, ...] = field(default_factory=tuple)
    epsilon: float = 1e-6

    @classmethod
    def from_dict(cls, config: dict[str, object] | object | None) -> "NVFP4Config":
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if isinstance(config, dict):
            return cls(
                block_size=int(config.get("block_size", 16)),
                quantize_activations=bool(config.get("quantize_activations", False)),
                skip_modules=tuple(config.get("skip_modules", ())),
                epsilon=float(config.get("epsilon", 1e-6)),
            )

        return cls(
            block_size=int(getattr(config, "block_size", 16)),
            quantize_activations=bool(getattr(config, "quantize_activations", False)),
            skip_modules=tuple(getattr(config, "skip_modules", ())),
            epsilon=float(getattr(config, "epsilon", 1e-6)),
        )


def _fold_last_dim(
    x: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, int, int]:
    original_last_dim = x.shape[-1]
    pad = (block_size - original_last_dim % block_size) % block_size
    if pad > 0:
        x = F.pad(x, (0, pad))
    return x.reshape(-1, x.shape[-1]), original_last_dim, pad


def _unfold_last_dim(
    x: torch.Tensor, prefix_shape: torch.Size, original_last_dim: int, pad: int
) -> torch.Tensor:
    if pad > 0:
        x = x[..., :original_last_dim]
    return x.reshape(*prefix_shape, original_last_dim)


def _closest_fp4_values(x: torch.Tensor) -> torch.Tensor:
    levels = FP4_QUANT_LEVELS.to(device=x.device)
    bucket_indices = torch.bucketize(x, levels)
    left_indices = torch.clamp(bucket_indices - 1, min=0, max=levels.numel() - 1)
    right_indices = torch.clamp(bucket_indices, min=0, max=levels.numel() - 1)
    left = levels[left_indices]
    right = levels[right_indices]
    choose_right = (x - left).abs() > (right - x).abs()
    return torch.where(choose_right, right, left)


def cast_to_nvfp4(
    x: torch.Tensor, block_size: int = 16, epsilon: float = 1e-6
) -> torch.Tensor:
    original_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    flat, original_last_dim, pad = _fold_last_dim(x_fp32, block_size)
    blocks = flat.reshape(flat.shape[0], -1, block_size)

    # block scale normalized against FP4 max level 6.0 to use all 16 levels (matches nvfp4_pack.py)
    block_scales = blocks.abs().amax(dim=-1) / FP4_E2M1_PEAK
    global_scale = torch.clamp(block_scales.amax(), min=epsilon) / FP8_E4M3_PEAK
    global_scale = torch.clamp(global_scale, min=epsilon)

    fp8_scales = torch.clamp(block_scales / global_scale, min=0.0, max=FP8_E4M3_PEAK)
    if hasattr(torch, "float8_e4m3fn"):
        fp8_scales = fp8_scales.to(torch.float8_e4m3fn).to(torch.float32)

    real_scales = torch.clamp(fp8_scales * global_scale, min=epsilon)
    normalized = blocks / real_scales.unsqueeze(-1)
    quantized = _closest_fp4_values(normalized) * real_scales.unsqueeze(-1)

    restored = _unfold_last_dim(
        quantized.reshape(flat.shape[0], -1),
        x_fp32.shape[:-1],
        original_last_dim,
        pad,
    )
    return restored.to(original_dtype)


def pseudo_quantize_to_nvfp4(
    x: torch.Tensor, block_size: int = 16, epsilon: float = 1e-6
) -> torch.Tensor:
    quantized = cast_to_nvfp4(x, block_size=block_size, epsilon=epsilon)
    return x + (quantized - x).detach()


class NVFP4Linear(nn.Linear):
    def __init__(self, linear: nn.Linear, config: NVFP4Config):
        super().__init__(
            linear.in_features, linear.out_features, bias=linear.bias is not None
        )
        self.quant_config = config
        self.quantization_enabled = True
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone())

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: NVFP4Config) -> "NVFP4Linear":
        return cls(linear=linear, config=config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.quantization_enabled:
            return F.linear(inputs, self.weight, self.bias)

        quantized_weight = pseudo_quantize_to_nvfp4(
            self.weight,
            block_size=self.quant_config.block_size,
            epsilon=self.quant_config.epsilon,
        )
        if self.quant_config.quantize_activations:
            inputs = pseudo_quantize_to_nvfp4(
                inputs,
                block_size=self.quant_config.block_size,
                epsilon=self.quant_config.epsilon,
            )
        return F.linear(inputs, quantized_weight, self.bias)


def _module_excluded(module_path: str, skip_modules: Iterable[str]) -> bool:
    return any(skip_name and skip_name in module_path for skip_name in skip_modules)


def enable_nvfp4_quantization(
    module: nn.Module, config: NVFP4Config, prefix: str = ""
) -> list[str]:
    replaced_modules: list[str] = []
    for child_name, child in list(module.named_children()):
        module_path = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, NVFP4Linear):
            continue
        if isinstance(child, nn.Linear) and not _module_excluded(
            module_path, config.skip_modules
        ):
            setattr(module, child_name, NVFP4Linear.from_linear(child, config))
            replaced_modules.append(module_path)
            continue
        replaced_modules.extend(
            enable_nvfp4_quantization(child, config, prefix=module_path)
        )
    return replaced_modules


def toggle_nvfp4_quantization(module: nn.Module, enabled: bool) -> int:
    updated_modules = 0
    for child in module.modules():
        if isinstance(child, NVFP4Linear):
            child.quantization_enabled = enabled
            updated_modules += 1
    return updated_modules
