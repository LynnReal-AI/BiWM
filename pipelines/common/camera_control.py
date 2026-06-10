# Camera control (definition + preparation) — shared across backbones.
# The discrete camera-action design and Plucker handling here are referenced from
# the HunyuanWorld-1.5 implementation.
# (feat: build_camera_inputs add skip_plucker + GPU plucker path)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple
import numpy as np



# =============================================================================
# Utility Functions
# =============================================================================





def custom_meshgrid(*args):
    """torch.meshgrid with 'ij' indexing (torch>=2.0.0)."""
    return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device):
    """Plucker embedding computation (from wan_video_camera_controller.py / CameraCtrl).

    Args:
        K: [B, V, 4] intrinsic vectors [fx, fy, cx, cy] in pixel space
        c2w: [B, V, 4, 4] camera-to-world matrices
        H: pixel height
        W: pixel width
        device: torch device

    Returns:
        plucker: [B, V, H, W, 6] Plucker coordinates (o×d, d)
    """
    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B, V, 1

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, HW, 3
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, HW, 3
    rays_dxo = torch.linalg.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    return plucker


class _ResidualBlock(nn.Module):
    """Residual block for WanCameraAdapter (same as Wan's CameraControlAdapter)."""
    def __init__(self, dim, zero_init: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        if zero_init:
            nn.init.zeros_(self.conv1.weight)
            nn.init.zeros_(self.conv2.weight)
        else:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class WanCameraAdapter(nn.Module):
    """Pixel-resolution Plucker -> token-level camera features.

    Aligned with VideoX-Fun pipeline architecture:
        Temporal PixelUnshuffle → Spatial PixelUnshuffle → strided Conv2d → ResidualBlock → output_scale

    Time dimension handling (VideoX-Fun convention):
        - Input: [B, 6, F_pixel, H, W] at full pixel frame rate
        - Temporal unshuffle: repeat first frame S times, then reshape
          [B, 6, F_pixel, H, W] → [B, 6*S, F_latent, H, W]
        - This is LOSSLESS: no interpolation, all pixel-frame camera info preserved

    For Wan 2.2 (VAE temporal_stride=4, spatial_stride=32 = VAE 16x + patch 2x):
        - Temporal unshuffle: 6 * 4 = 24 channels, F_pixel → F_latent frames
        - Spatial PixelUnshuffle(8): 24 * 64 = 1536 channels
        - Conv2d(k=4, s=4): 1536 → 3072 (out_dim), 4x spatial downsample
        - Total spatial: 8 * 4 = 32x (matches VAE 16x + patch 2x)

    Zero-initialization: output_scale starts at 0 (same as ref), so adapter
    outputs zero at init, preserving base model behavior.
    """

    def __init__(self, in_channels: int = 6, out_dim: int = 3072,
                 downscale_factor: int = 8, conv_kernel: int = 4, conv_stride: int = 4,
                 num_residual_blocks: int = 1, vae_temporal_stride: int = 4,
                 split_plucker: bool = False,
                 output_scale_init: float = 0.0,
                 disable_output_scale: bool = False,
                 zero_init: bool = True):
        """Wan 2.2 defaults: spatial 8*4=32x (VAE 16x + patch 2x), temporal 4x."""
        super().__init__()
        self.out_dim = out_dim
        self.in_channels = in_channels
        self.downscale_factor = downscale_factor
        self.conv_stride = conv_stride
        self.vae_temporal_stride = vae_temporal_stride
        self.total_spatial_ratio = downscale_factor * conv_stride  # 4 * 4 = 16 for Wan
        self.split_plucker = split_plucker
        self.zero_init = zero_init

        init_mode = "ZERO" if zero_init else "Xavier"

        # After temporal unshuffle: in_channels * vae_temporal_stride (6 * 4 = 24)
        ch_temporal = in_channels * vae_temporal_stride  # 48


        # Original single-branch path
        # After spatial PixelUnshuffle: ch_temporal * downscale_factor^2 (48 * 64 = 3072)
        ch_after_unshuffle = ch_temporal * downscale_factor ** 2  # 3072

        # Architecture: Spatial PixelUnshuffle → Conv2d (strided) → ResidualBlock
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv = nn.Conv2d(ch_after_unshuffle, out_dim,
                                kernel_size=conv_kernel, stride=conv_stride, padding=0)
        if zero_init:
            nn.init.zeros_(self.conv.weight)
        else:
            nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.residual_blocks = nn.Sequential(
            *[_ResidualBlock(out_dim, zero_init=zero_init) for _ in range(num_residual_blocks)]
        )

        # Learnable output scale (init=0 for stable training start)
        if not disable_output_scale:
            self.output_scale = nn.Parameter(torch.full((1,), output_scale_init))
        else:
            self.output_scale = None

        # Init summary logging
        total_params = sum(p.numel() for p in self.parameters())
        all_zero = all(p.data.abs().max().item() == 0.0 for p in self.parameters())
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                w_abs_mean = module.weight.data.abs().mean().item()
                w_max = module.weight.data.abs().max().item()
                layer_type = "Conv2d" if isinstance(module, nn.Conv2d) else "Linear"

    def forward(
        self,
        plucker_pixel: torch.Tensor,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        """
        Args:
            plucker_pixel: [B, 6, F_pixel, H_pixel, W_pixel] at full pixel frame rate
            latent_frames, latent_height, latent_width: target latent dimensions

        Returns:
            [B, F_lat*H_lat*W_lat, out_dim]
        """
        B, C, F_in, H_in, W_in = plucker_pixel.shape
        S = self.vae_temporal_stride  # 8
        r = self.total_spatial_ratio  # 32



        # Step 1: Temporal pixel unshuffle
        first_repeated = plucker_pixel[:, :, 0:1].repeat(1, 1, S, 1, 1)  # [B, 6, 4, 544, 960]
        rest = plucker_pixel[:, :, 1:]  # [B, 6, 120, 544, 960]
        x = torch.cat([first_repeated, rest], dim=2)  # [B, 6, 124, 544, 960]

        F_after_unshuffle = x.shape[2]  # total frame count after repeating the first frame S times + concatenating the rest
        F_needed = latent_frames * S   # target frame count = latent_frames * vae_stride

        # Align to F_needed: truncate the front segment or repeat the last frame
        if F_after_unshuffle > F_needed:
            x = x[:, :, :F_needed]
        elif F_after_unshuffle < F_needed:
            pad_n = F_needed - F_after_unshuffle
            x = torch.cat([x, x[:, :, -1:].expand(-1, -1, pad_n, -1, -1)], dim=2)

        F_lat = latent_frames  # ensure consistency with latent_frames

        # Reshape: [B, 6, 124, H, W] → [B, 24, 31, H, W]
        x = x.view(B, C, F_lat, S, H_in, W_in)       # [B, 6, 31, 4, H, W]
        x = x.permute(0, 1, 3, 2, 4, 5)               # [B, 6, 4, 31, H, W]
        x = x.reshape(B, C * S, F_lat, H_in, W_in)    # [B, 24, 31, H, W]

        ch = C * S  # 24

        # Step 2: Spatial alignment (ensure H, W match latent * total_spatial_ratio)
        H_target = latent_height * r
        W_target = latent_width * r
        _spatial_interp = (H_in != H_target or W_in != W_target)
        if _spatial_interp:
            try:
                x = F.interpolate(
                    x.float(),
                    size=(F_lat, H_target, W_target),
                    mode='trilinear',
                    align_corners=True,
                ).to(dtype=plucker_pixel.dtype)
                H_in, W_in = H_target, W_target
            except Exception as e:
                raise

        # Step 3: Per-frame spatial PixelUnshuffle + Conv2d + ResBlock
        T = F_lat
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, ch, H_in, W_in)  # [B*31, 24, 544, 960]

        x = self.pixel_unshuffle(x)    # [B*31, 1536, 68, 120]   (24*8²=1536, H/8, W/8)
        x = self.conv(x)               # [B*31, 3072, 17, 30]    (1536→3072, k=4, s=4)
        x = self.residual_blocks(x)    # [B*31, 3072, 17, 30]

        # Reshape back: [B*31, 3072, 17, 30] → [B, 3072, 31, 17, 30]
        x = x.view(B, T, self.out_dim, latent_height, latent_width)
        x = x.permute(0, 2, 1, 3, 4)  # [B, 3072, 31, 34, 60]

        # Learnable output scale (init=0 for stable training)
        if self.output_scale is not None:
            x = x * self.output_scale

        # Flatten: [B, 3072, 31, 17, 30] → [B, 15810, 3072]
        x = x.flatten(2).transpose(1, 2)

        return x


# HunyuanVideo action mapping: one-hot [4] -> label 0~8
HUNYUAN_ACTION_MAPPING = {
    (0, 0, 0, 0): 0,  # static
    (1, 0, 0, 0): 1,  # forward
    (0, 1, 0, 0): 2,  # backward
    (0, 0, 1, 0): 3,  # right
    (0, 0, 0, 1): 4,  # left
    (1, 0, 1, 0): 5,  # forward + right
    (1, 0, 0, 1): 6,  # forward + left
    (0, 1, 1, 0): 7,  # backward + right
    (0, 1, 0, 1): 8,  # backward + left
}


def _hunyuan_one_hot_to_label(one_hot: np.ndarray) -> int:
    key = tuple(int(x) for x in one_hot)
    return HUNYUAN_ACTION_MAPPING.get(key, 0)


def c2w_to_action_labels(
    c2w_matrices: np.ndarray,
    vae_temporal_stride: int = 4,
    translation_threshold: float = 0.0001,
    rotation_threshold: float = 0.001,
) -> torch.Tensor:
    """Convert c2w to 81-class action labels (9 trans × 9 rot).

    Args:
        c2w_matrices: [N, 4, 4] extrinsics (pixel-frame level)
        vae_temporal_stride: VAE temporal downsampling stride (Wan=4)
        translation_threshold: translation threshold
        rotation_threshold: rotation threshold

    Returns:
        action_labels: [latent_num] tensor (0~80)
    """
    from scipy.spatial.transform import Rotation as R

    num_frames = len(c2w_matrices)
    stride = vae_temporal_stride
    latent_num = (num_frames - 1) // stride + 1

    latent_frame_indices = [min(i * stride, num_frames - 1) for i in range(latent_num)]
    c2ws_sampled = np.array([c2w_matrices[i] for i in latent_frame_indices])

    trans_one_hot = np.zeros((latent_num, 4), dtype=np.int32)
    rotate_one_hot = np.zeros((latent_num, 4), dtype=np.int32)

    # frame-to-frame relative pose
    relative_c2w = np.zeros_like(c2ws_sampled)
    relative_c2w[0] = c2ws_sampled[0]
    if len(c2ws_sampled) > 1:
        C_inv = np.linalg.inv(c2ws_sampled[:-1])
        relative_c2w[1:] = np.einsum('nij,njk->nik', C_inv, c2ws_sampled[1:])

    # adaptive translation threshold (based on 5% of the median step size)
    move_thresh = translation_threshold
    rot_thresh_deg = rotation_threshold * (180.0 / np.pi)  # convert to degrees
    if latent_num > 1:
        step_norms = np.linalg.norm(relative_c2w[1:, :3, 3], axis=1)
        nonzero = step_norms[step_norms > 1e-12]
        if nonzero.size > 0:
            move_thresh = max(1e-6, 0.05 * np.median(nonzero))

    for i in range(1, latent_num):
        move_dirs = relative_c2w[i, :3, 3]
        move_norms = np.linalg.norm(move_dirs)

        if move_norms > move_thresh:
            move_norm_dirs = move_dirs / move_norms
            trans_angles_deg = np.degrees(np.arccos(np.clip(move_norm_dirs, -1.0, 1.0)))
    trans_labels = np.array([_hunyuan_one_hot_to_label(trans_one_hot[i]) for i in range(latent_num)])
    rotate_labels = np.array([_hunyuan_one_hot_to_label(rotate_one_hot[i]) for i in range(latent_num)])
    action_labels = trans_labels * 9 + rotate_labels
    return torch.tensor(action_labels, dtype=torch.long)


def build_camera_inputs(
    K_ctrl: torch.Tensor,
    c2w_ctrl: torch.Tensor,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    device: torch.device,
    dtype: torch.dtype,
    mode: str = "default",
    pixel_height: int = 0,
    pixel_width: int = 0,
    original_height: float = 0.0,
    original_width: float = 0.0,
    compute_action_labels: bool = True,
    c2w_raw: torch.Tensor = None,  # original extrinsics (un-normalized), used for 81-class classification
    discrete_mode: str = "81cls",  # "81cls" (81-class labels) or "13dim" (13-dim velocity)
    skip_plucker: bool = False,  # skip Plucker computation (saves CPU + copy when no_continuous_camera)
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Prepare camera control inputs from dataset outputs.

    The dataset has already done the preprocessing:
    1. Intrinsics normalized to [0,1] (fx_n, fy_n, cx_n≈0.5, cy_n≈0.5)
    2. Extrinsics already passed through normalize_cam_c2w (first frame = identity, max(||t||) ≈ 1.0)

    This function is responsible for:
    - Scaling the normalized intrinsics to the target pixel resolution via aspect-ratio correction (wan-style process_pose_file)
    - Calling ray_condition to compute Plücker coordinates (batched, wan convention)
    - Optionally computing discrete action labels

    Args:
        K_ctrl: normalized [0,1] intrinsics [3,3] or [B,3,3]
        c2w_ctrl: normalized extrinsics [N,4,4] or [B,N,4,4] (first frame is identity)
        latent_frames, latent_height, latent_width: Latent dimensions
        device, dtype: Target device and dtype
        mode: 'wan_inject' (pixel-res Plucker)
        pixel_height, pixel_width: Pixel resolution for Plucker computation
        original_height, original_width: original video resolution (for intrinsic aspect-ratio correction)
        compute_action_labels: whether to compute discrete action labels (default True).
            Set to False to skip the scipy dependency and action classification, returning action_labels=None.

    Returns:
        (action_labels, plucker_coords, action_vectors)
        action_labels: [B, latent_num] tensor (0~80) or None
        plucker_coords: [B, 6, F, H_pixel, W_pixel]
        action_vectors: [B, latent_num, 8] float tensor (8-dim velocity) or None
    """
    K_np = K_ctrl.cpu().numpy() if isinstance(K_ctrl, torch.Tensor) else K_ctrl
    c2w_np = c2w_ctrl.cpu().numpy() if isinstance(c2w_ctrl, torch.Tensor) else c2w_ctrl

    if K_np.ndim == 2:
        K_np = K_np[np.newaxis]
    if c2w_np.ndim == 2:
        c2w_np = c2w_np[np.newaxis]
    if c2w_np.ndim == 3 and K_np.ndim == 3 and K_np.shape[0] != c2w_np.shape[0]:
        K_np = np.broadcast_to(K_np, (c2w_np.shape[0],) + K_np.shape[-2:])
    if c2w_np.ndim == 4:
        batch_size = c2w_np.shape[0]
    else:
        c2w_np = c2w_np[np.newaxis]
        batch_size = 1
    if K_np.ndim == 2:
        K_np = K_np[np.newaxis]

    action_labels_list = [] if (compute_action_labels and discrete_mode == "81cls") else None
    action_vectors_list = [] if (compute_action_labels and discrete_mode != "81cls") else None
    K_vec_list = []     # [B, V, 4] for ray_condition
    c2w_all_list = []   # [B, V, 4, 4] for ray_condition

    H, W = pixel_height, pixel_width

    for b in range(batch_size):
        c2w_b = c2w_np[b]
        K_b = K_np[b] if K_np.shape[0] > 1 else K_np[0]
        num_frames = c2w_b.shape[0]

        # Discrete action (two modes)
        if compute_action_labels:
            # 81-class: original un-normalized c2w + max-norm + threshold
            if c2w_raw is not None:
                c2w_raw_np = c2w_raw.cpu().numpy() if isinstance(c2w_raw, torch.Tensor) else c2w_raw
                c2w_raw_b = c2w_raw_np[b] if c2w_raw_np.ndim == 4 else (c2w_raw_np if c2w_raw_np.ndim == 3 else c2w_b)
            else:
                c2w_raw_b = c2w_b
            _al = c2w_to_action_labels(c2w_raw_b, vae_temporal_stride=4)
            action_labels_list.append(_al)

        # The dataset has already done normalize_cam_c2w (first frame=identity),
        # so use the original c2w_b directly, without a second relativization
        c2w_for_plucker = c2w_b.astype(np.float32)

        # ── wan-style intrinsic handling: normalized [0,1] → pixel-space K vector [V, 4] ──
        fx_n = float(K_b[0, 0])
        fy_n = float(K_b[1, 1])
        cx_n = float(K_b[0, 2])
        cy_n = float(K_b[1, 2])

        # aspect-ratio correction (wan process_pose_file convention)
        if original_width > 0 and original_height > 0:
            sample_wh_ratio = W / H
            pose_wh_ratio = original_width / original_height
            if pose_wh_ratio > sample_wh_ratio:
                resized_ori_w = H * pose_wh_ratio
                fx_n = resized_ori_w * fx_n / W
            else:
                resized_ori_h = W / pose_wh_ratio
                fy_n = resized_ori_h * fy_n / H

        # scale to target pixel resolution: [fx*W, fy*H, cx*W, cy*H]
        K_vec = np.array([fx_n * W, fy_n * H, cx_n * W, cy_n * H], dtype=np.float32)
        K_vec_frames = np.tile(K_vec[None, :], (num_frames, 1))  # [V, 4]
        K_vec_list.append(K_vec_frames)
        c2w_all_list.append(c2w_for_plucker)


    # ── Batched Plucker via ray_condition (wan convention) ──
    # : add skip_plucker early return, otherwise compute directly on GPU (originally device='cpu' + final .to(gpu))
    if skip_plucker:
        plucker_coords = None
    else:
        K_batch = torch.as_tensor(np.stack(K_vec_list, axis=0), dtype=torch.float32, device=device)   # [B, V, 4] on GPU
        c2w_batch = torch.as_tensor(np.stack(c2w_all_list, axis=0), dtype=torch.float32, device=device)  # [B, V, 4, 4] on GPU
        plucker_bhw6 = ray_condition(K_batch, c2w_batch, H, W, device=device)  # [B, V, H, W, 6] on GPU
        plucker_coords = plucker_bhw6.permute(0, 4, 1, 2, 3).contiguous()    # [B, 6, V, H, W]
        plucker_coords = plucker_coords.to(dtype=dtype)  # already on device, only convert dtype

    # Action post-processing
    action_labels = None
    action_vectors = None
    if compute_action_labels:
        try:
            if discrete_mode == "81cls" and action_labels_list:
                action_labels = torch.stack(action_labels_list, dim=0).to(device=device)
                al_len = action_labels.shape[1]
                if al_len != latent_frames and al_len > 0 and latent_frames > 0:
                    resample_idx = torch.linspace(0, al_len - 1, latent_frames).long()
                    action_labels = action_labels[:, resample_idx]
            elif action_vectors_list:
                action_vectors = torch.stack(action_vectors_list, dim=0).to(device=device, dtype=dtype)
                av_len = action_vectors.shape[1]
                if av_len != latent_frames and av_len > 0 and latent_frames > 0:
                    resample_idx = torch.linspace(0, av_len - 1, latent_frames).long()
                    action_vectors = action_vectors[:, resample_idx]
        except Exception:
            action_labels = None
            action_vectors = None

    return action_labels, plucker_coords, action_vectors