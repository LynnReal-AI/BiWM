# BiWM Stage 3: offline NVFP4 weight export. Converts BF16 weights -> packed-FP4 (uint8, 2 values/byte)
# + FP8-E4M3 block scale + global scale, as LightX2V-compatible safetensors loaded by nvfp4_kernel.py.
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


FP4_E2M1_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
FP8_E4M3_PEAK = 448.0


@dataclass(slots=True)
class NVFP4PackConfig:
    enabled: bool = False
    block_size: int = 16
    epsilon: float = 1e-6
    include_generator: bool = True
    include_generator_ema: bool = True
    skip_keys: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, config: dict | object | None) -> "NVFP4PackConfig":
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if isinstance(config, dict):
            return cls(
                enabled=bool(config.get("enabled", False)),
                block_size=int(config.get("block_size", 16)),
                epsilon=float(config.get("epsilon", 1e-6)),
                include_generator=bool(config.get("include_generator", True)),
                include_generator_ema=bool(config.get("include_generator_ema", True)),
                skip_keys=tuple(config.get("skip_keys", ())),
            )

        return cls(
            enabled=bool(getattr(config, "enabled", False)),
            block_size=int(getattr(config, "block_size", 16)),
            epsilon=float(getattr(config, "epsilon", 1e-6)),
            include_generator=bool(getattr(config, "include_generator", True)),
            include_generator_ema=bool(getattr(config, "include_generator_ema", True)),
            skip_keys=tuple(getattr(config, "skip_keys", ())),
        )


def _round_up_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _into_blocked_layout(row_major: torch.Tensor) -> torch.Tensor:
    orig_rows, orig_cols = row_major.shape
    n_row_blocks = _round_up_div(orig_rows, 128)
    n_col_blocks = _round_up_div(orig_cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = torch.zeros(
        (padded_rows, padded_cols),
        dtype=row_major.dtype,
        device=row_major.device,
    )
    padded[:orig_rows, :orig_cols] = row_major
    x = padded.reshape(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    x = x.reshape(-1, 4, 32, 4).transpose(1, 2)
    x = x.reshape(-1, 32, 16)
    return x.reshape(padded_rows, padded_cols).contiguous()


def _closest_fp4_indices(x: torch.Tensor) -> torch.Tensor:
    lut = FP4_E2M1_TABLE.to(device=x.device)
    distances = (x.unsqueeze(-1) - lut).abs()
    return distances.argmin(dim=-1).to(torch.uint8)


def cast_weight_to_packed_nvfp4(
    weight: torch.Tensor,
    block_size: int = 16,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(
            f"NVFP4 export expects 2D weights, got shape {tuple(weight.shape)}"
        )
    if block_size != 16:
        raise ValueError(
            "Only block_size=16 is supported for block16-packed NVFP4 export"
        )

    weight_fp32 = weight.detach().to(torch.float32)
    out_features, in_features = weight_fp32.shape
    num_scale_cols = _round_up_div(in_features, block_size)
    padded_in_features = num_scale_cols * block_size

    if padded_in_features != in_features:
        weight_fp32 = F.pad(weight_fp32, (0, padded_in_features - in_features))

    blocks = weight_fp32.reshape(out_features, num_scale_cols, block_size)
    raw_scales = blocks.abs().amax(dim=-1) / 6.0
    max_scale = raw_scales.max()

    if float(max_scale.item()) <= epsilon:
        global_scale = torch.tensor(1.0, dtype=torch.float32)
        fp8_scales_quantized = torch.zeros_like(raw_scales, dtype=torch.float32)
        real_scales = torch.ones_like(raw_scales, dtype=torch.float32)
    else:
        global_scale = torch.clamp(max_scale / FP8_E4M3_PEAK, min=epsilon).to(
            torch.float32
        )
        fp8_scales = torch.clamp(raw_scales / global_scale, min=0.0, max=FP8_E4M3_PEAK)
        if hasattr(torch, "float8_e4m3fn"):
            fp8_scales_quantized = fp8_scales.to(torch.float8_e4m3fn).to(torch.float32)
        else:
            fp8_scales_quantized = fp8_scales
        real_scales = torch.where(
            fp8_scales_quantized > 0,
            fp8_scales_quantized * global_scale,
            torch.ones_like(fp8_scales_quantized, dtype=torch.float32),
        )

    normalized_blocks = blocks / real_scales.unsqueeze(-1)
    fp4_indices = _closest_fp4_indices(normalized_blocks).reshape(
        out_features, padded_in_features
    )
    packed_weight = (
        (fp4_indices[:, 0::2].to(torch.int32) * 16)
        | fp4_indices[:, 1::2].to(torch.int32)
    ).to(torch.uint8)

    if hasattr(torch, "float8_e4m3fn"):
        scale_tensor = _into_blocked_layout(fp8_scales_quantized.to(torch.float8_e4m3fn))
    else:
        scale_tensor = _into_blocked_layout(fp8_scales_quantized.to(torch.float32))

    return (
        packed_weight.cpu().contiguous(),
        scale_tensor.cpu().contiguous(),
        global_scale.cpu().contiguous(),
    )


def transform_state_dict_to_nvfp4(
    state_dict: dict[str, torch.Tensor],
    config: NVFP4PackConfig,
) -> dict[str, torch.Tensor]:
    exported_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        tensor = value.detach().cpu().contiguous()
        is_quantizable = (
            key.endswith(".weight")
            and tensor.ndim == 2
            and tensor.is_floating_point()
            and not any(skip_key and skip_key in key for skip_key in config.skip_keys)
        )
        if not is_quantizable:
            exported_state_dict[key] = tensor
            continue

        prefix = key[: -len(".weight")]
        packed_weight, weight_scale, weight_scale_2 = cast_weight_to_packed_nvfp4(
            tensor,
            block_size=config.block_size,
            epsilon=config.epsilon,
        )
        exported_state_dict[key] = packed_weight
        exported_state_dict[f"{prefix}.weight_scale"] = weight_scale
        exported_state_dict[f"{prefix}.weight_scale_2"] = weight_scale_2

    return exported_state_dict

