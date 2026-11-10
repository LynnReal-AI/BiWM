"""H3 Stage 1 packing: Qwen(first RGB, caption) | anchor | target video.

Adapted from SolarWM stage0.5_v7. Positions stay float64 until native H3 RoPE.
"""
import math

import torch


def patchify(latents):
    """[B,C,T,H,W] -> [B,T*(H/2)*(W/2),4*C], native H3 ordering."""
    b, c, t, h, w = latents.shape
    if h % 2 or w % 2:
        raise ValueError("H3 latent spatial dimensions must be divisible by 2")
    return (latents.reshape(b, c, t, h // 2, 2, w // 2, 2)
            .permute(0, 2, 3, 5, 1, 4, 6).reshape(b, -1, c * 4).contiguous())


def unpatchify(rows, frames, height, width):
    b, _, channels = rows.shape
    return (rows.reshape(b, frames, height // 2, width // 2, channels // 4, 2, 2)
            .permute(0, 4, 1, 2, 5, 3, 6).reshape(b, channels // 4, frames, height, width))


def build_layout(text_tags, frames, height, width):
    """Native nonuniform H3 clock; anchor shares target frame 0's RoPE/pose."""
    if frames < 1 or height % 2 or width % 2:
        raise ValueError("Invalid H3 latent grid")
    device = text_tags.device
    text = len(text_tags)
    rows = (height // 2) * (width // 2)
    count = text + (frames + 1) * rows
    positions = torch.zeros(count, 3, dtype=torch.float64, device=device)
    positions[:text, 0] = torch.arange(text, device=device)
    axes = []
    for size in (height, width):
        ratio = size / math.sqrt(height * width)
        axes.append(((1 - ratio) / 2 + ratio * torch.arange(
            size // 2, dtype=torch.float64, device=device) / (size // 2)) * 32)
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 2)
    # Each five latent frames advance by RGB spans [1,4,4,4,4] * 5/3.
    spans = torch.tensor([1, 4, 4, 4, 4], dtype=torch.float64, device=device)
    spans = spans[torch.arange(frames, device=device) % 5] * (5 / 3)
    times = torch.cat((spans.new_zeros(1), spans.cumsum(0)[:-1])) + text
    positions[text:text + rows, 0] = times[0]
    positions[text:text + rows, 1:] = grid
    positions[text + rows:, 0] = times.repeat_interleave(rows)
    positions[text + rows:, 1:] = grid.repeat(frames, 1)
    video_indices = torch.arange(text, count, device=device)
    token_tags = torch.cat((text_tags.long(), torch.zeros(
        (frames + 1) * rows, dtype=torch.long, device=device)))
    return dict(position_ids=positions, token_tags=token_tags,
                text_indices=torch.arange(text, device=device),
                video_indices=video_indices,
                audio_indices=torch.empty(0, dtype=torch.long, device=device)), rows


def row_timesteps(layout, target_t, anchor_rows):
    """Qwen/target use native clean-time t; the slightly noisy anchor uses .999."""
    row_times = target_t.expand(len(layout["token_tags"])).clone()
    start = len(layout["text_indices"])
    row_times[start:start + anchor_rows] = .999
    return torch.unique(row_times, sorted=True, return_inverse=True)
