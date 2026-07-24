# >>> BiWM Stage 2 — history compression: FramePack <<<
"""
FramePack history compression — unified interface.

Compress the already-generated history latent [B, 48, T, H, W] into mem tokens via
"pyramid multi-scale spatio-temporal downsampling": nearby frames are lightly compressed
(preserving detail), distant frames are heavily compressed (saving tokens). Returns:
    (mem_tokens [B, N, dim],  mem_indices_grid [B, 3, N, 2])
— this unified interface can be fed directly into the
history_kv_tokens / history_indices_grid path of wan/modules/model.py
(inside model, each token's (start,end) bounds are floored to their midpoint → RoPE positions).

The multi-scale conv is a learnable parameter, initialized by trilinear upsampling of the
main patch_embedding, and trained end-to-end together with DMD.

Coordinate convention (RoPE position uses patch-grid units):
  - Main patch_embedding kernel/stride = (1,2,2): latent → patch-grid spatial base stride = 2.
  - framepack spatial scale s∈{1,2,4,8,16}: conv uses stride=(1, 2s, 2s) (latent units) for downsampling,
    but the RoPE position step = s (patch-grid units). I.e. s=1 tokens land at patch coords (0,1,2,...),
    fully aligned with the target blocks; s=2/4/... distant tokens cover s patch cells.
  - Time: T position uses the latent frame index (= patch-T, T stride=1), continuous with target gen_t;
    only the 32x distant segment first goes through 2x_f (T stride=2) frame compression (1 token covers 2 latent frames).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Compression strategies (adaptively selected by history frame count; segment order = farthest to nearest)
# ============================================================
@dataclass
class _Span:
    """A single compression segment: name is for debugging only; (f0,f1) is a frame slice (supports negative index/None);
    scale is the spatial compression factor; when frame_compress=True (32x only) first apply 2x frame compression then 16x spatial compression."""
    name: str
    f0: int
    f1: Optional[int]
    scale: int
    frame_compress: bool = False


@dataclass
class _Plan:
    name: str
    max_frames: int
    segs: List[_Span] = field(default_factory=list)


# Equivalent to yume COMPRESSION_STRATEGIES (same frame ranges/factors), only the dataclass fields are more concise and clear.
_STRATEGIES: List[_Plan] = [
    _Plan("tiny", 6, [
        _Span("sink",   0,  1,    1),
        _Span("mid",    1, -1,    2),
        _Span("near",  -1,  None, 1),
    ]),
    _Plan("small", 22, [
        _Span("sink",   0,  1,    1),
        _Span("far",    1, -5,    4),
        _Span("mid",   -5, -3,    2),
        _Span("near",  -3,  None, 1),
    ]),
    _Plan("medium", 86, [
        _Span("sink",    0,   1,    1),
        _Span("far",     1, -21,    8),
        _Span("mid",   -21,  -5,    4),
        _Span("near",   -5,  -3,    2),
        _Span("recent", -3,   None, 1),
    ]),
    _Plan("large", 342, [
        _Span("sink",    0,    1,    2),
        _Span("far",     1,  -85,   16),
        _Span("mid",   -85,  -21,    8),
        _Span("near",  -21,   -5,    4),
        _Span("recent", -5,   -3,    2),
        _Span("current",-3,    None, 1),
    ]),
    _Plan("xlarge", 1366, [
        _Span("sink",     0,    1,    2),
        _Span("far",      1, -341,   32, frame_compress=True),
        _Span("mid",   -341,  -85,   16),
        _Span("near",   -85,  -21,    8),
        _Span("recent", -21,   -5,    4),
        _Span("close",   -5,   -3,    2),
        _Span("current", -3,    None, 1),
    ]),
]


def _choose_plan(num_frames: int) -> _Plan:
    for st in _STRATEGIES:
        if num_frames <= st.max_frames:
            return st
    return _STRATEGIES[-1]


def _determine_slice(f0: int, f1: Optional[int], total: int) -> Tuple[int, int]:
    """Resolve a frame slice (supporting negative index/None) into [start, end) absolute coordinates (clamped to [0,total])."""
    start = f0 + total if f0 < 0 else f0
    end = total if f1 is None else (f1 + total if f1 < 0 else f1)
    start = max(0, min(start, total))
    end = max(0, min(end, total))
    return start, end


# ============================================================
# Multi-scale learnable conv construction (initialized by trilinear upsampling of the main patch_embedding)
# ============================================================
def _build_scaled_conv(base: nn.Conv3d, kernel: Tuple[int, int, int]) -> nn.Conv3d:
    """Build a downsampling conv with kernel=stride=kernel from the base patch_embedding (Conv3d),
    its weight obtained by trilinear interpolation of base.weight to the target kernel (same method as yume upsample_conv3d_weights), learnable."""
    out_ch, in_ch = base.weight.shape[0], base.weight.shape[1]
    w = F.interpolate(base.weight.data.float(), size=kernel, mode="trilinear", align_corners=False)
    conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel, stride=kernel, padding=0,
                     bias=base.bias is not None)
    conv.weight.data = w.to(base.weight.dtype)
    if base.bias is not None:
        conv.bias.data = base.bias.data.clone()
    return conv


def _build_frame_conv(in_ch: int, dtype, device) -> nn.Conv3d:
    """2x frame compression conv (in=out=in_ch, kernel/stride=(2,1,1)), initialized as temporal mean pooling (0.5,0.5)."""
    conv = nn.Conv3d(in_ch, in_ch, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        for c in range(in_ch):
            conv.weight[c, c, 0, 0, 0] = 0.5
            conv.weight[c, c, 1, 0, 0] = 0.5
    return conv.to(dtype=dtype, device=device)


def _pad_space(x: torch.Tensor, mult: int) -> torch.Tensor:
    """Zero-pad H,W on the right to an integer multiple of mult (for strided conv, same semantics as yume convpad)."""
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (mult - h % mult) % mult
    pad_w = (mult - w % mult) % mult
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))   # (W_left,W_right,H_left,H_right)
    return x


class FramePackPastCompressor(nn.Module):
    """FramePack history compressor with a unified memory-token interface.

    forward(history_latent [B,48,T,H,W]) -> (mem_tokens [B,N,dim], mem_indices_grid [B,3,N,2])
    """

    # Spatial scale → conv's (kt,kh,kw); base spatial stride=2, so scale s → 2s.
    _SPATIAL_SCALES = (1, 2, 4, 8, 16)

    def __init__(self, base_patch_embedding: nn.Conv3d, dim: int,
                 enable_frame_compression: bool = True):
        super().__init__()
        self.dim = dim
        self.in_channels = base_patch_embedding.weight.shape[1]
        self.enable_frame_compression = enable_frame_compression
        _dtype = base_patch_embedding.weight.dtype
        _device = base_patch_embedding.weight.device

        # Multi-scale spatial conv (learnable). Keys use strings to avoid the ModuleDict integer-key restriction.
        self.spatial = nn.ModuleDict()
        for s in self._SPATIAL_SCALES:
            self.spatial[str(s)] = _build_scaled_conv(base_patch_embedding, (1, 2 * s, 2 * s))
        # 2x frame compression conv used by the 32x distant segment (in=out=in_channels)
        self.frame_2x = _build_frame_conv(self.in_channels, _dtype, _device)

    # ---- Single-segment compression: returns (tokens [B, n, dim], bounds [3, n, 2]) ----
    def _compress_segment(self, frames: torch.Tensor, seg: _Span,
                          frame_start_abs: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """frames: [B, C, Lf, H, W] raw frames of this segment; frame_start_abs: absolute frame index of this segment's first frame within the whole history."""
        B = frames.shape[0]
        is_fc = seg.frame_compress and self.enable_frame_compression
        t_stride = 2 if is_fc else 1                       # number of original latent frames covered by 1 output frame
        spatial_scale = 16 if seg.scale == 32 else seg.scale
        # conv spatial stride in latent units = 2·scale (main patch base stride=2), but the RoPE position
        # uses patch-grid units, so position step = scale.
        sp_conv_stride = 2 * spatial_scale                  # conv stride (latent units, used only for padding)
        sp_pos = spatial_scale                              # position step (patch-grid units)

        x = frames
        if is_fc:                                           # first apply 2x frame compression
            x = self.frame_2x(x)                            # [B,C,Lf2,H,W]
            if x.shape[2] == 0:
                return None
        x = _pad_space(x, sp_conv_stride)
        x = self.spatial[str(spatial_scale)](x)            # [B, dim, T', H', W']

        _, _, Tp, Hp, Wp = x.shape
        if Tp == 0 or Hp == 0 or Wp == 0:
            return None
        tokens = x.flatten(2).transpose(1, 2).contiguous()  # [B, T'*H'*W', dim]

        # ---- Integer (T,H,W) bounds in latent units ----
        dev = x.device
        t_idx = torch.arange(Tp, device=dev, dtype=torch.float32)
        t_start = frame_start_abs + t_idx * t_stride
        t_end = t_start + t_stride
        h_idx = torch.arange(Hp, device=dev, dtype=torch.float32)
        h_start = h_idx * sp_pos
        h_end = h_start + sp_pos
        w_idx = torch.arange(Wp, device=dev, dtype=torch.float32)
        w_start = w_idx * sp_pos
        w_end = w_start + sp_pos

        Ts, Hs, Ws = torch.meshgrid(t_start, h_start, w_start, indexing="ij")
        Te, He, We = torch.meshgrid(t_end, h_end, w_end, indexing="ij")
        starts = torch.stack([Ts.flatten(), Hs.flatten(), Ws.flatten()], dim=0)  # [3, n]
        ends = torch.stack([Te.flatten(), He.flatten(), We.flatten()], dim=0)
        bounds = torch.stack([starts, ends], dim=-1)                              # [3, n, 2]
        return tokens, bounds

    def forward(self, history_latent: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """history_latent: [B, 48, T, H, W] → (mem_tokens [B,N,dim], mem_indices_grid [B,3,N,2])."""
        assert history_latent.dim() == 5, f"expect [B,C,T,H,W], got {list(history_latent.shape)}"
        B, C, T, H, W = history_latent.shape
        strat = _choose_plan(T)

        all_tokens: List[torch.Tensor] = []
        all_bounds: List[torch.Tensor] = []
        for seg in strat.segs:
            s, e = _determine_slice(seg.f0, seg.f1, T)
            if s >= e:
                continue
            seg_frames = history_latent[:, :, s:e]
            out = self._compress_segment(seg_frames, seg, frame_start_abs=s)
            if out is None:
                continue
            tok, bnd = out
            all_tokens.append(tok)
            all_bounds.append(bnd)

        if not all_tokens:                                  # fallback for empty history (theoretically not triggered, only called when b>0)
            dev = history_latent.device
            empty_tok = history_latent.new_zeros(B, 0, self.dim)
            empty_bnd = torch.zeros(B, 3, 0, 2, device=dev, dtype=torch.float32)
            return empty_tok, empty_bnd

        mem_tokens = torch.cat(all_tokens, dim=1)                         # [B, N, dim]
        mem_bounds = torch.cat(all_bounds, dim=1)                         # [3, N, 2]
        mem_indices_grid = mem_bounds.unsqueeze(0).expand(B, -1, -1, -1).contiguous()  # [B,3,N,2]
        return mem_tokens, mem_indices_grid


def make_framepack_compressor(args, transformer: nn.Module,
                               dtype=torch.bfloat16) -> FramePackPastCompressor:
    """Factory: build the FramePack compressor from the main transformer's patch_embedding (Conv3d in=48,out=dim,stride=(1,2,2)).
    Must be called before FSDP wrap (consistent with make_history_encoder)."""
    base = transformer.patch_embedding
    comp = FramePackPastCompressor(
        base_patch_embedding=base,
        dim=transformer.dim,
        enable_frame_compression=not getattr(args, "framepack_disable_frame_compression", False),
    )
    comp = comp.to(dtype=dtype, device=base.weight.device)
    n_params = sum(p.numel() for p in comp.parameters())
    print(f"[FramePack] built: in={comp.in_channels}, dim={comp.dim}, "
          f"spatial_scales={comp._SPATIAL_SCALES}, params={n_params/1e6:.1f}M "
          f"(multi-scale learnable conv, trilinear-init from patch_embedding)")
    return comp
