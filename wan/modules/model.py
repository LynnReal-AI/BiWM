# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']


# 81-class action → text mapping (9 trans × 9 rot); action_label = trans_label * 9 + rot_label
TRANS_TEXT = [
    "Camera does not move.",                     # 0
    "Camera moves forward.",                     # 1
    "Camera moves backward.",                    # 2
    "Camera moves right.",                       # 3
    "Camera moves left.",                        # 4
    "Camera moves forward-right.",               # 5
    "Camera moves forward-left.",                # 6
    "Camera moves backward-right.",              # 7
    "Camera moves backward-left.",               # 8
]

ROT_TEXT = [
    "Camera does not rotate.",                   # 0
    "Camera pitches up.",                        # 1
    "Camera pitches down.",                      # 2
    "Camera yaws right.",                        # 3
    "Camera yaws left.",                         # 4
    "Camera pitches up and yaws right.",         # 5
    "Camera pitches up and yaws left.",          # 6
    "Camera pitches down and yaws right.",       # 7
    "Camera pitches down and yaws left.",        # 8
]


def build_action_text_table() -> List[str]:
    """Build 81 action description strings (trans_label * 9 + rot_label)."""
    table = []
    for t_idx in range(9):
        for r_idx in range(9):
            table.append(f"{TRANS_TEXT[t_idx]} {ROT_TEXT[r_idx]}")
    return table


ACTION_TEXT_TABLE = build_action_text_table()  # len=81


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs, positions=None):
    """Rotary position embedding apply.

    Args:
        x: [B, seq_len, num_heads, head_dim]
        grid_sizes: [B, 3] (F, H, W) — raster scan over the first f*h*w tokens; the rest is padding.
        freqs: [1024, head_dim//2] complex
        positions: Optional [B, seq_len_eff, 3] explicit per-token (T_idx, H_idx, W_idx); when not
                   None it overrides the raster scan, looking up freqs directly by positions[i, j].
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i in range(x.size(0)):
        if positions is not None:
            # ★ Explicit per-token positions path
            pos_i = positions[i]  # [seq_len_eff, 3]
            seq_len = pos_i.size(0)
            t_idx = pos_i[:, 0].long().clamp_(0, freqs[0].size(0) - 1)
            h_idx = pos_i[:, 1].long().clamp_(0, freqs[1].size(0) - 1)
            w_idx = pos_i[:, 2].long().clamp_(0, freqs[2].size(0) - 1)
            # each token independently takes (T,H,W) freq → concat → [seq_len, dim]
            freqs_i = torch.cat([
                freqs[0][t_idx],   # [seq_len, c - 2*(c//3)]
                freqs[1][h_idx],   # [seq_len, c//3]
                freqs[2][w_idx],   # [seq_len, c//3]
            ], dim=-1).reshape(seq_len, 1, -1)
        else:
            # original raster scan path
            f, h, w = grid_sizes[i].tolist()
            seq_len = f * h * w
            freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(seq_len, 1, -1)

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, positions=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            positions: Optional [B, seq_len_eff, 3] per-token (T,H,W) for explicit RoPE
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs, positions=positions),
            k=rope_apply(k, grid_sizes, freqs, positions=positions),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        per_frame_context=None,
        per_frame_context_lens=None,
        tokens_per_frame=None,
        num_latent_frames=None,
        positions=None,           # explicit per-token (T,H,W) for packforcing_pretrain RoPE
        n_mem=0,                  # number of history mem prefix tokens (for rollout camera control)
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            per_frame_context(Tensor, optional): [B*T, S_cam+S_cap, C] per-frame context
            per_frame_context_lens(Tensor, optional): [B*T] valid lengths
            tokens_per_frame(int, optional): H_patch * W_patch
            num_latent_frames(int, optional): T (= number of target latent frames, excluding mem prefix)
            positions(Tensor, optional): [B, seq_len_eff, 3] per-token (T,H,W) indices.
                                          packforcing_pretrain mode uses history_indices_grid + Ω real indices.
            n_mem(int, optional): ★ rollout camera control: number of history mem prefix tokens at the start of the sequence.
                                  These n_mem tokens use caption-only context for cross-attn;
                                  the following T*tpf target tokens use per_frame_context (cam-text).
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs, positions=positions)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn
        def cross_attn_ffn(x, context, context_lens, e):
            if per_frame_context is not None and tokens_per_frame is not None:
                # Per-frame cross-attention over [mem prefix (n_mem) | target (T*tpf) | pad]:
                #   mem segment → caption-only; target segment → per-frame cam-text + caption.
                B = x.shape[0]
                T = num_latent_frames                 # number of target latent frames (excluding mem)
                tpf = tokens_per_frame
                L = x.shape[1]

                x_norm = self.norm3(x)
                ca_out = torch.zeros_like(x_norm)

                # --- mem segment: caption-only cross-attn ---
                if n_mem > 0:
                    mem_part = x_norm[:, :n_mem, :]
                    ca_out[:, :n_mem, :] = self.cross_attn(mem_part, context, context_lens)

                # --- target segment: per-frame cam-text cross-attn ---
                tgt_lo = n_mem
                tgt_hi = n_mem + T * tpf
                tgt_part = x_norm[:, tgt_lo:tgt_hi, :].reshape(B * T, tpf, -1)
                tgt_ca = self.cross_attn(tgt_part, per_frame_context, per_frame_context_lens)
                ca_out[:, tgt_lo:tgt_hi, :] = tgt_ca.reshape(B, T * tpf, -1)

                x = x + ca_out
            else:
                x = x + self.cross_attn(self.norm3(x), context, context_lens)

            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module=None, value=False, enable=None):
        if enable is not None:
            value = enable
        self.gradient_checkpointing = value

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        self.gradient_checkpointing = False

        # initialize weights
        self.init_weights()

    # =================================================================
    # Camera Control
    # =================================================================
    def add_camera_control_parameters(self, num_actions: int = 81):
        """Add discrete camera control: HunyuanActionEmbedder (81-class action → timestep modulation)."""
        from wan.modules.action_embedder import HunyuanActionEmbedder
        self.hycam_action_embedder = HunyuanActionEmbedder(
            hidden_size=self.dim, freq_dim=self.freq_dim, num_actions=num_actions,
        )
        dp = sum(p.numel() for p in self.hycam_action_embedder.parameters())
        print(f"[WanModel-Camera] action embedder (zero-init-out): {dp:,} params")

    def precompute_cam_text_embeddings(self, encode_fn, device, dtype=torch.bfloat16):
        """Pre-encode 81 action texts using text encoder and store as raw buffer.

        Raw embeddings (text_dim space) are stored; projection through
        self.text_embedding happens in forward() after concatenation with caption.

        Args:
            encode_fn: text encoder function, takes List[str] → List[Tensor[S, text_dim]]
            device: target device
            dtype: target dtype
        """
        action_texts = ACTION_TEXT_TABLE  # 81 strings
        raw_embs = encode_fn(action_texts)  # List of Tensor[S_i, text_dim=4096]
        self._cam_text_raw = [e.to(device=device, dtype=dtype) for e in raw_embs]
        self._use_cam_text_cross_attn = True
        # fixed-length padded version [num_actions, S_cam_max, text_dim] for vectorized injection;
        # cross-attn is order-independent, so this is numerically equivalent to per-frame concat.
        _S = max(int(e.shape[0]) for e in self._cam_text_raw)
        _td = int(self._cam_text_raw[0].shape[1])
        _pad = torch.zeros(len(self._cam_text_raw), _S, _td, device=device, dtype=dtype)
        for _i, _e in enumerate(self._cam_text_raw):
            _pad[_i, :_e.shape[0]] = _e
        self._cam_text_padded = _pad                                       # [num_actions, S_cam_max, td]
        print(f"[CamText] Stored {len(self._cam_text_raw)} raw embeddings, text_dim={self._cam_text_raw[0].shape[1]}")

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        # Camera control kwargs
        action_labels=None,
        cond_latent_frames=0,          # I2V/V2V: number of condition frames (sigma=0 when >0)
        # Frame Preservation history concat: caller passes HR encoder output (history_kv_tokens) +
        # LR latent (history_lr_latent); LR is patchified here, concat with HR and prepended.
        history_kv_tokens=None,        # [B, N_hr, dim] HR encoder output (from WanVideoHistoryEncoder)
        history_indices_grid=None,     # [B, 3, N_hr, 2] HR grid (T,H,W) start/end, used for the RoPE midpoint
        history_lr_latent=None,        # [B, 48, T_lr, H_lr, W_lr] LR latent (separate LR pixel → VAE encode)
        gen_t_indices_override=None,   # [K] target Ω real indices (for strict RoPE)
    ):
        r"""
        Forward pass through the diffusion model.

        Args:
            x (List[Tensor]):  [C_in, F, H, W] each
            t (Tensor):  [B]
            context (List[Tensor]):  [L, C] each
            seq_len (int):  max sequence length (must include history N_mem)
            y (List[Tensor], optional):  conditional video for i2v
            action_labels (Tensor, optional):  [B, T_lat] discrete actions
            cond_latent_frames (int):  number of condition frames, when >0 condition frames have sigma=0
            history_kv_tokens (Tensor, optional):  [B, N_mem, dim] history encoder output, prepended before the token sequence
            history_indices_grid (Tensor, optional):  [B, 3, N_mem] (T,H,W) position of each mem token within the history latent
            gen_t_indices_override (Tensor, optional):  [K] packforcing_pretrain Ω indices
        """
        if self.model_type == 'i2v':
            assert y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # patch embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        # Frame Preservation history concat: HR mem = history_kv_tokens, LR mem = patchified
        # history_lr_latent; concat into mem_final and prepend to the token sequence (sliced off before head).
        _N_mem = 0
        positions = None
        if history_kv_tokens is not None:
            assert history_kv_tokens.shape[-1] == self.dim, (
                f"history_kv_tokens dim {history_kv_tokens.shape[-1]} != self.dim {self.dim}"
            )
            hr_mem = history_kv_tokens   # [B, N_hr, dim]

            # LR branch: reuse self.patch_embedding on the separate LR latent; HR/LR fused via concat
            # (no shape constraint). total N_mem = N_hr + N_lr.
            if history_lr_latent is not None:
                assert history_lr_latent.shape[1] == self.in_dim, (
                    f"history_lr_latent channel {history_lr_latent.shape[1]} != in_dim {self.in_dim}"
                )
                # self.patch_embedding is nn.Conv3d(in_dim=48, dim=3072, kernel=(1,2,2), stride=(1,2,2))
                lr_patchified = self.patch_embedding(history_lr_latent.to(self.patch_embedding.weight.dtype))
                # → [B, dim, T_lr, H_lr, W_lr]
                _, _, T_lr_p, H_lr_p, W_lr_p = lr_patchified.shape
                lr_mem = lr_patchified.flatten(2).transpose(1, 2)   # [B, N_lr, dim]
                # concat fusion, no shape match required
                mem_final = torch.cat([hr_mem, lr_mem], dim=1)      # [B, N_hr + N_lr, dim]
            else:
                mem_final = hr_mem
            _N_mem = mem_final.shape[1]
            # === build positions: [B, N_mem + N_target, 3] ===
            B = len(x)
            assert B == 1, "packforcing_pretrain v1 currently only supports B=1 (consistent with the training loop)"
            T_t, H_t, W_t = grid_sizes[0].tolist()    # target latent grid
            # mem positions: history_indices_grid is [B, 3, N_mem, 2] (start, end), take the floored midpoint
            assert history_indices_grid is not None, (
                "packforcing_pretrain mode must pass history_indices_grid (HistoryEncoder._build_indices_grid output)"
            )
            assert history_indices_grid.dim() == 4 and history_indices_grid.shape[1] == 3, (
                f"history_indices_grid shape should be [B, 3, N_mem, 2], actual {list(history_indices_grid.shape)}"
            )
            hig = history_indices_grid.to(device=x[0].device)
            mem_mid = ((hig[..., 0] + hig[..., 1]) / 2.0).floor().long()    # [B, 3, N_hr]
            hr_mem_pos = mem_mid.permute(0, 2, 1).contiguous()               # [B, N_hr, 3] (T,H,W)

            # LR mem positions: T = history latent index; spatial = raster scan after LR patchify
            if history_lr_latent is not None:
                _, _, T_lr_p, H_lr_p, W_lr_p = lr_patchified.shape
                t_lr = torch.arange(T_lr_p, dtype=torch.long, device=x[0].device)
                h_lr = torch.arange(H_lr_p, dtype=torch.long, device=x[0].device)
                w_lr = torch.arange(W_lr_p, dtype=torch.long, device=x[0].device)
                T_g, H_g, W_g = torch.meshgrid(t_lr, h_lr, w_lr, indexing='ij')
                lr_mem_pos = torch.stack([T_g.flatten(), H_g.flatten(), W_g.flatten()], dim=-1)
                lr_mem_pos = lr_mem_pos.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, N_lr, 3]
                mem_pos = torch.cat([hr_mem_pos, lr_mem_pos], dim=1)         # [B, N_hr+N_lr, 3]
            else:
                mem_pos = hr_mem_pos

            # target positions: T uses the Ω real indices (if gen_t_indices_override is given), otherwise 0..T_t-1
            if gen_t_indices_override is not None:
                assert gen_t_indices_override.shape[0] == T_t, (
                    f"gen_t_indices_override len {gen_t_indices_override.shape[0]} != target T {T_t}"
                )
                t_idx_target = gen_t_indices_override.to(dtype=torch.long, device=x[0].device)
            else:
                t_idx_target = torch.arange(T_t, dtype=torch.long, device=x[0].device)
            h_idx = torch.arange(H_t, dtype=torch.long, device=x[0].device)
            w_idx = torch.arange(W_t, dtype=torch.long, device=x[0].device)
            T_g, H_g, W_g = torch.meshgrid(t_idx_target, h_idx, w_idx, indexing='ij')
            target_pos = torch.stack([T_g.flatten(), H_g.flatten(), W_g.flatten()], dim=-1)  # [N_target, 3]
            target_pos = target_pos.unsqueeze(0).expand(B, -1, -1).contiguous()              # [B, N_target, 3]

            positions = torch.cat([mem_pos, target_pos], dim=1)              # [B, N_mem + N_target, 3]

            # prepend mem_final (HR + LR) before latent tokens
            x = [torch.cat([mem_final[i:i+1].to(u.dtype), u], dim=1) for i, u in enumerate(x)]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len, (
            f"actual seq_len {seq_lens.max().item()} > declared seq_len {seq_len}. "
            f"packforcing_pretrain mode: the caller must add history N_mem ({_N_mem}) into seq_len."
        )
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if cond_latent_frames > 0:
            # I2V/V2V separated timestep: condition frame tokens t=0, generated frame tokens t=t[0]
            F_lat = grid_sizes[0, 0].item()
            H_p = grid_sizes[0, 1].item()
            W_p = grid_sizes[0, 2].item()
            tokens_per_frame = H_p * W_p
            cond_latent_frames = min(cond_latent_frames, F_lat - 1)  # keep at least 1 generated frame
            cond_tokens = cond_latent_frames * tokens_per_frame
            gen_tokens = (F_lat - cond_latent_frames) * tokens_per_frame
            per_token_t = torch.cat([
                torch.zeros(cond_tokens, device=t.device, dtype=t.dtype),
                t[0].expand(gen_tokens),
            ])
            with torch.amp.autocast('cuda', dtype=torch.float32):
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, per_token_t).float())
                e = e.unsqueeze(0)  # [1, L_actual, dim]
                if e.shape[1] < seq_len:
                    e = torch.cat([e, e.new_zeros(1, seq_len - e.shape[1], e.shape[2])], dim=1)
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
        elif _N_mem > 0:
            # packforcing_pretrain/rollout — the mem prefix is a clean condition,
            # it must use timestep=0 (cannot share σ with target for AdaLN modulation);
            # target latent tokens use the real σ. So in e0, the mem segment=t0 and the target segment=σ.
            _sigma_scalar = t.flatten()[0]   # target σ*1000 (scalar)
            per_token_t = torch.cat([
                torch.zeros(_N_mem, device=t.device, dtype=t.dtype),          # mem: t=0
                _sigma_scalar.expand(seq_len - _N_mem),                        # target(+pad): σ
            ])
            with torch.amp.autocast('cuda', dtype=torch.float32):
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, per_token_t).float()
                ).unsqueeze(0)   # [1, seq_len, dim]
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
        else:
            if t.dim() == 1:
                t = t.expand(t.size(0), seq_len)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                bt = t.size(0)
                t_flat = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            t_flat).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # --- Per-frame camera text cross-attention ---
        per_frame_context = None
        per_frame_context_lens = None
        _tokens_per_frame = None
        _num_latent_frames = None

        _use_cam_text = (getattr(self, '_use_cam_text_cross_attn', False)
                         and action_labels is not None
                         and hasattr(self, '_cam_text_raw'))
        if _use_cam_text:
            B = len(context) if isinstance(context, list) else context.shape[0]
            T = grid_sizes[0, 0].item()  # num latent frames
            H_p = grid_sizes[0, 1].item()
            W_p = grid_sizes[0, 2].item()
            _tokens_per_frame = H_p * W_p
            _num_latent_frames = T

            al = action_labels
            if al.shape[1] != T:
                al = F.interpolate(
                    al.float().unsqueeze(1), size=T, mode='nearest'
                ).squeeze(1).long()
            assert al.min() >= 0 and al.max() < len(self._cam_text_raw), \
                f"action_labels out of range: min={al.min().item()}, max={al.max().item()}, num_actions={len(self._cam_text_raw)}"

            text_dim_raw = self._cam_text_raw[0].shape[1]  # text_dim (4096)
            S_target = self.text_len  # 512

            # vectorized: cam gather _cam_text_padded[al] → [B,T,S_cam,td]; caption broadcast to T
            # frames; fixed-length [cam | caption | zeros] truncated to S_target (order-independent).
            _cam = self._cam_text_padded[al]                                  # [B, T, S_cam, td]
            _S_cam = _cam.shape[2]
            _S_capmax = max(int(u.shape[0]) for u in context)
            _cap = torch.stack([torch.cat([u, u.new_zeros(_S_capmax - u.shape[0], text_dim_raw)])
                                for u in context])                           # [B, S_capmax, td]
            _cap = _cap[:, None].expand(B, T, _S_capmax, text_dim_raw)        # [B, T, S_capmax, td]
            _comb = torch.cat([_cam, _cap], dim=2).reshape(B * T, _S_cam + _S_capmax, text_dim_raw)
            if _comb.shape[1] >= S_target:
                per_frame_raw = _comb[:, :S_target].contiguous()             # [B*T, S_target, td]
            else:
                per_frame_raw = torch.cat(
                    [_comb, _comb.new_zeros(B * T, S_target - _comb.shape[1], text_dim_raw)], dim=1)

            # cam + caption projected together for the target segment per-frame cross-attn
            per_frame_context = self.text_embedding(per_frame_raw)  # [B*T, S_target, dim]

            # rollout (mem prefix): mem segment uses caption-only context, project an extra copy here
            if _N_mem > 0:
                context = self.text_embedding(
                    torch.stack([
                        torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                        for u in context
                    ]))

        else:
            # context (regular caption projection)
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))

        context_lens=None
        per_frame_context_lens = None
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            per_frame_context=per_frame_context,
            per_frame_context_lens=per_frame_context_lens,
            tokens_per_frame=_tokens_per_frame,
            num_latent_frames=_num_latent_frames,
            positions=positions,        # explicit RoPE for packforcing_pretrain
            n_mem=_N_mem,               # mem prefix length, cross_attn_ffn uses it to segment
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        block, x, use_reentrant=False, **kwargs)
                else:
                    x = block(x, **kwargs)

        # in packforcing_pretrain mode, slice off the history mem prefix
        # head + unpatchify only process the target latent tokens
        if _N_mem > 0:
            x = x[:, _N_mem:, :]
            # e/e0 must be sliced in sync (head uses e for AdaLN, lengths must match)
            if e.dim() == 3 and e.shape[1] >= _N_mem:
                e = e[:, _N_mem:, :]

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
