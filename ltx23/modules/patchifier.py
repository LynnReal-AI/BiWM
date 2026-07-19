"""Video latent patchification and coordinate conversion for LTX-Video 2.3."""
import math
from typing import NamedTuple, Optional, Tuple

import einops
import torch


# =============================================================================
# 类型定义
# =============================================================================

class SpatioTemporalScaleFactors(NamedTuple):
    """时空下采样因子"""
    time: int
    width: int
    height: int

    @classmethod
    def default(cls) -> "SpatioTemporalScaleFactors":
        return cls(time=8, width=32, height=32)


VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class VideoLatentShape(NamedTuple):
    """视频 latent 形状: (batch, channels, frames, height, width)"""
    batch: int
    channels: int
    frames: int
    height: int
    width: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.height, self.width])

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "VideoLatentShape":
        return VideoLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            height=shape[3],
            width=shape[4],
        )

    def mask_shape(self) -> "VideoLatentShape":
        return self._replace(channels=1)


# =============================================================================
# Video Patchifier
# =============================================================================

class VideoLatentPatchifier:
    """视频 latent patchifier"""

    def __init__(self, patch_size: int = 1):
        self._patch_size = (1, patch_size, patch_size)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    def get_token_count(self, tgt_shape: VideoLatentShape) -> int:
        return math.prod(tgt_shape.to_torch_shape()[2:]) // math.prod(self._patch_size)

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        """将 [B, C, F, H, W] 转换为 [B, F*H*W, C*p1*p2*p3]"""
        latents = einops.rearrange(
            latents,
            "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
            p1=self._patch_size[0],
            p2=self._patch_size[1],
            p3=self._patch_size[2],
        )
        return latents

    def unpatchify(self, latents: torch.Tensor, output_shape: VideoLatentShape) -> torch.Tensor:
        """将 [B, F*H*W, C*p*q] 转换回 [B, C, F, H, W]"""
        assert self._patch_size[0] == 1, "Temporal patch size must be 1"

        patch_grid_frames = output_shape.frames // self._patch_size[0]
        patch_grid_height = output_shape.height // self._patch_size[1]
        patch_grid_width = output_shape.width // self._patch_size[2]

        latents = einops.rearrange(
            latents,
            "b (f h w) (c p q) -> b c f (h p) (w q)",
            f=patch_grid_frames,
            h=patch_grid_height,
            w=patch_grid_width,
            p=self._patch_size[1],
            q=self._patch_size[2],
        )
        return latents

    def get_patch_grid_bounds(
        self,
        output_shape: VideoLatentShape,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """获取每个 patch 的边界坐标 [batch, 3, num_patches, 2]"""
        frames = output_shape.frames
        height = output_shape.height
        width = output_shape.width
        batch_size = output_shape.batch

        grid_coords = torch.meshgrid(
            torch.arange(start=0, end=frames, step=self._patch_size[0], device=device),
            torch.arange(start=0, end=height, step=self._patch_size[1], device=device),
            torch.arange(start=0, end=width, step=self._patch_size[2], device=device),
            indexing="ij",
        )

        patch_starts = torch.stack(grid_coords, dim=0)
        patch_size_delta = torch.tensor(
            self._patch_size,
            device=patch_starts.device,
            dtype=patch_starts.dtype,
        ).view(3, 1, 1, 1)

        patch_ends = patch_starts + patch_size_delta
        latent_coords = torch.stack((patch_starts, patch_ends), dim=-1)

        latent_coords = einops.repeat(
            latent_coords,
            "c f h w bounds -> b c (f h w) bounds",
            b=batch_size,
            bounds=2,
        )
        return latent_coords


# =============================================================================
# 坐标计算
# =============================================================================

def get_pixel_coords(
    latent_coords: torch.Tensor,
    scale_factors: SpatioTemporalScaleFactors,
    causal_fix: bool = False,
) -> torch.Tensor:
    """
    将 latent 坐标转换为像素坐标

    Args:
        latent_coords: [batch, 3, num_patches, 2] latent 边界
        scale_factors: 时空缩放因子
        causal_fix: 是否修正因果编码

    Returns:
        [batch, 3, num_patches, 2] 像素坐标
    """
    broadcast_shape = [1] * latent_coords.ndim
    broadcast_shape[1] = -1
    scale_tensor = torch.tensor(scale_factors, device=latent_coords.device).view(*broadcast_shape)

    pixel_coords = latent_coords * scale_tensor

    if causal_fix:
        pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + 1 - scale_factors[0]).clamp(min=0)

    return pixel_coords
