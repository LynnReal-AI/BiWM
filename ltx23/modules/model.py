# Copyright 2024-2025 LTX-Video 2.3
"""LTX-Video 2.3 transformer used by the BiWM training pipelines."""
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from typing import Optional, Tuple
from ltx23.modules.attention import AttentionFunction, AttentionCallable
from ltx23.modules.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx23.modules.timestep_embedding import get_timestep_embedding, TimestepEmbedding
from ltx23.modules.rope import apply_rotary_emb, LTXRopeType, precompute_freqs_cis, generate_freq_grid_np, generate_freq_grid_pytorch
from pipelines.utils.context_parallel import (
    cp_is_active as is_cp_enabled,
    fetch_cp_world_size as get_cp_world_size,
    disperse_sequence as scatter_sequence,
    collect_for_loss as gather_for_loss,
    apply_ulysses_attention,
    pad_for_cp_divisible as pad_to_cp_divisible,
    strip_cp_padding as unpad_from_cp,
)

__all__ = [
    'LTX23Model', 'LTX23ModelType', 'VideoLatentShape', 'X0Model', 'to_denoised', 'Modality',
    'Attention', 'FeedForward', 'LTXRopeType', 'precompute_freqs_cis', 'rms_norm',
]


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


# =============================================================================
# AdaLN coefficient helper (LTX 2.3)
# =============================================================================
ADALN_NUM_BASE_PARAMS = 6
ADALN_NUM_CROSS_ATTN_PARAMS = 3


def adaln_embedding_coefficient(cross_attention_adaln: bool) -> int:
    return ADALN_NUM_BASE_PARAMS + (ADALN_NUM_CROSS_ATTN_PARAMS if cross_attention_adaln else 0)


@dataclass(frozen=True)
class Modality:
    latent: torch.Tensor       # (B, T, D)
    sigma: torch.Tensor        # (B,) - current sigma for cross-attention timestep (NEW in 2.3)
    timesteps: torch.Tensor    # (B, T)
    positions: torch.Tensor    # (B, 3, T) for video
    context: torch.Tensor
    enabled: bool = True
    context_mask: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None  # (B, T, T) self-attention mask (NEW in 2.3)


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False       # NEW in 2.3
    cross_attention_adaln: bool = False       # NEW in 2.3


class PixArtAlphaTextProjection(torch.nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int | None = None, act_fn: str = "gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = torch.nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = torch.nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = torch.nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = torch.nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Timesteps(torch.nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = get_timestep_embedding(
            timesteps, self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class PixArtAlphaCombinedTimestepSizeEmbeddings(torch.nn.Module):
    def __init__(self, embedding_dim: int, size_emb_dim: int):
        super().__init__()
        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_dtype: torch.dtype) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        return timesteps_emb


class AdaLayerNormSingle(torch.nn.Module):
    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6):
        super().__init__()
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim, size_emb_dim=embedding_dim // 3)
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True)

    def forward(self, timestep: torch.Tensor, hidden_dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class VideoLatentShape:
    def __init__(self, batch_size, num_frames, num_channels, height, width):
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.height = height
        self.width = width

    def to_tuple(self):
        return (self.batch_size, self.num_frames, self.num_channels, self.height, self.width)


class LTX23ModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTX23ModelType.AudioVideo, LTX23ModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTX23ModelType.AudioVideo, LTX23ModelType.AudioOnly)


class GELUApprox(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.proj(x), approximate="tanh")


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, dim_out: int, mult: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELUApprox(dim, inner_dim)
        self.net = torch.nn.Sequential(project_in, torch.nn.Identity(), torch.nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Attention with Gated Attention support (LTX 2.3)
# =============================================================================

class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = attention_function

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        # Per-head gating (NEW in 2.3)
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        context = x if context is None else context
        use_attention = not all_perturbed

        v = self.to_v(context)

        if not use_attention:
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(context)

            q = self.q_norm(q)
            k = self.k_norm(k)

            if pe is not None:
                q = apply_rotary_emb(q, pe, self.rope_type)
                k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

            out = self.attention_function(q, k, v, self.heads, mask)

            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        # Per-head gating (NEW in 2.3)
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            b, t, _ = out.shape
            out = out.view(b, t, self.heads, self.dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits)  # zero-init → identity (2*0.5=1.0)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, self.heads * self.dim_head)

        return self.to_out(out)


class LTX23SelfAttention(torch.nn.Module):
    """Self-attention with context parallel support."""
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = attention_function

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        # Per-head gating (NEW in 2.3)
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        is_self_attn = context is None
        context = x if context is None else context
        use_attention = not all_perturbed

        v = self.to_v(context)

        if not use_attention:
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(context)

            q = self.q_norm(q)
            k = self.k_norm(k)

            if pe is not None:
                q = apply_rotary_emb(q, pe, self.rope_type)
                k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

            if is_self_attn and is_cp_enabled():
                if not hasattr(self, '_ulysses_debug_cnt'):
                    self._ulysses_debug_cnt = 0
                self._ulysses_debug_cnt += 1
                if self._ulysses_debug_cnt <= 3:
                    import sys
                    _r = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                    print(f"[Ulysses-Debug][rank={_r}] 进入 all-to-all, q={q.shape}, heads={self.heads}", flush=True)
                out = apply_ulysses_attention(q, k, v, self.heads, self.attention_function, mask)
                if self._ulysses_debug_cnt <= 3:
                    print(f"[Ulysses-Debug][rank={_r}] all-to-all 完成, out={out.shape}", flush=True)
            else:
                out = self.attention_function(q, k, v, self.heads, mask)

            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        # Per-head gating (NEW in 2.3)
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            b, t, _ = out.shape
            out = out.view(b, t, self.heads, self.dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, self.heads * self.dim_head)

        return self.to_out(out)


class LTX23CrossAttention(LTX23SelfAttention):
    pass


# =============================================================================
# Cross-Attention AdaLN helper (LTX 2.3)
# =============================================================================

def apply_cross_attention_adaln(
    x: torch.Tensor,
    context: torch.Tensor,
    attn,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor,
    context_mask: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    batch_size = x.shape[0]
    # prompt_timestep: (B, 1, 2*dim), prompt_scale_shift_table: (2, dim)
    shift_kv, scale_kv = (
        prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
        + prompt_timestep.reshape(batch_size, -1, 2, prompt_scale_shift_table.shape[-1])
    ).unbind(dim=2)
    attn_input = rms_norm(x, eps=norm_eps) * (1 + q_scale) + q_shift
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    return attn(attn_input, context=encoder_hidden_states, mask=context_mask) * q_gate


# =============================================================================
# LTX 2.3 Attention Block
# =============================================================================

class LTX23AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
        attention_function: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
    ):
        super().__init__()

        self.idx = idx
        self.cross_attention_adaln = video.cross_attention_adaln if video is not None else False

        self.attn1 = LTX23SelfAttention(
            query_dim=video.dim,
            heads=video.heads,
            dim_head=video.d_head,
            context_dim=None,
            rope_type=rope_type,
            norm_eps=norm_eps,
            attention_function=attention_function,
            apply_gated_attention=video.apply_gated_attention,
        )
        self.attn2 = LTX23CrossAttention(
            query_dim=video.dim,
            context_dim=video.context_dim,
            heads=video.heads,
            dim_head=video.d_head,
            rope_type=rope_type,
            norm_eps=norm_eps,
            attention_function=attention_function,
            apply_gated_attention=video.apply_gated_attention,
        )
        self.ff = FeedForward(video.dim, dim_out=video.dim)

        # AdaLN scale_shift_table: 6 (base) or 9 (with cross_attention_adaln)
        sst_size = adaln_embedding_coefficient(video.cross_attention_adaln)
        self.scale_shift_table = torch.nn.Parameter(torch.empty(sst_size, video.dim))

        # Cross-attention AdaLN: prompt scale/shift table (NEW in 2.3)
        if self.cross_attention_adaln:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video.dim))

        self.norm_eps = norm_eps

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]
        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def _apply_text_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply text cross-attention, with optional AdaLN modulation (LTX 2.3)."""
        if self.cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(
                self.scale_shift_table, x.shape[0], timesteps, slice(6, 9)
            )
            return apply_cross_attention_adaln(
                x, context, self.attn2,
                shift_q, scale_q, gate,
                self.prompt_scale_shift_table, prompt_timestep,
                context_mask, self.norm_eps,
            )
        return self.attn2(rms_norm(x, eps=self.norm_eps), context=context, mask=context_mask)

    def forward(
        self,
        x: torch.Tensor | None,
        timesteps: torch.Tensor | None = None,
        freqs: tuple[torch.Tensor, torch.Tensor] | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        perturbations: BatchedPerturbationConfig | None = None,
        prompt_timestep: Optional[torch.Tensor] = None,
        self_attention_mask: Optional[torch.Tensor] = None,
        per_frame_context: Optional[torch.Tensor] = None,
        tokens_per_frame: Optional[int] = None,
        num_latent_frames: Optional[int] = None,
    ) -> torch.Tensor:

        batch_size = x.shape[0] if x is not None else 1
        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        # Self-attention with AdaLN
        vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
            self.scale_shift_table, x.shape[0], timesteps, slice(0, 3)
        )

        norm_vx = rms_norm(x, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
        del vshift_msa, vscale_msa

        all_perturbed = perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx)
        none_perturbed = not perturbations.any_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx)
        v_mask = (
            perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, x)
            if not all_perturbed and not none_perturbed
            else None
        )
        x = (
            x
            + self.attn1(
                norm_vx,
                pe=freqs,
                mask=self_attention_mask,
                perturbation_mask=v_mask,
                all_perturbed=all_perturbed,
            )
            * vgate_msa
        )
        del vgate_msa, norm_vx, v_mask

        # Cross-attention (with optional AdaLN in 2.3)
        # 当 per_frame_context 存在时，使用逐帧 cross-attention (cam_text 模式)
        if per_frame_context is not None and tokens_per_frame is not None:
            B = x.shape[0]
            T = num_latent_frames
            tpf = tokens_per_frame
            if not hasattr(self, '_cam_text_ca_logged'):
                self._cam_text_ca_logged = True
                _adaln_tag = "with AdaLN" if self.cross_attention_adaln else "no AdaLN"
                print(f'[CamText-CrossAttn] 使用逐帧cross attn ({_adaln_tag}): x={list(x.shape)}, '
                      f'per_frame_context={list(per_frame_context.shape)}, B={B}, T={T}, tpf={tpf}')

            if self.cross_attention_adaln:
                # AdaLN query modulation: 与正常路径相同的 shift/scale/gate
                shift_q, scale_q, gate_q = self.get_ada_values(
                    self.scale_shift_table, B, timesteps, slice(6, 9)
                )
                # Per-token → per-frame reshape: [B, T*tpf, dim] → [B*T, tpf, dim]
                shift_q = shift_q[:, :T * tpf, :].reshape(B * T, tpf, -1)
                scale_q = scale_q[:, :T * tpf, :].reshape(B * T, tpf, -1)
                gate_q = gate_q[:, :T * tpf, :].reshape(B * T, tpf, -1)

                # KV modulation from prompt_timestep (context 端的 AdaLN)
                _pst_dim = self.prompt_scale_shift_table.shape[-1]
                shift_kv, scale_kv = (
                    self.prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
                    + prompt_timestep.reshape(B, -1, 2, _pst_dim)
                ).unbind(dim=2)
                # [B, 1, dim] → [B*T, 1, dim] (同 batch 内各帧共享)
                shift_kv = shift_kv.repeat_interleave(T, dim=0)
                scale_kv = scale_kv.repeat_interleave(T, dim=0)

                # 逐帧 cross-attention with AdaLN
                x_norm = rms_norm(x, eps=self.norm_eps)
                x_frames = x_norm[:, :T * tpf, :].reshape(B * T, tpf, -1)
                attn_input = x_frames * (1 + scale_q) + shift_q
                encoder_hidden_states = per_frame_context * (1 + scale_kv) + shift_kv
                ca_out = self.attn2(attn_input, context=encoder_hidden_states, mask=None) * gate_q
                ca_out = ca_out.reshape(B, T * tpf, -1)
            else:
                raise ValueError("Per-frame camera text requires cross-attention AdaLN")

            x = x + ca_out
        else:
            x = x + self._apply_text_cross_attention(
                x, context, timesteps, prompt_timestep, context_mask,
            )
            if not hasattr(self, '_normal_ca_logged'):
                self._normal_ca_logged = True
                print(f'[CamText-CrossAttn] 使用正常cross attn (非cam_text): context={list(context.shape) if context is not None else None}')

        # FFN with AdaLN
        vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
            self.scale_shift_table, x.shape[0], timesteps, slice(3, 6)
        )
        x_scaled = rms_norm(x, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
        x = x + self.ff(x_scaled) * vgate_mlp

        del vshift_mlp, vscale_mlp, vgate_mlp, x_scaled

        return x


# =============================================================================
# Main LTX 2.3 Model
# =============================================================================

class LTX23Model(ModelMixin, ConfigMixin):

    ignore_for_config = ['patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size']
    _no_split_modules = ['LTX23AttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='video_only',
                 num_attention_heads: int = 32,
                 attention_head_dim: int = 128,
                 in_channels: int = 128,
                 out_channels: int = 128,
                 num_layers: int = 48,
                 cross_attention_dim: int = 4096,
                 norm_eps: float = 1e-06,
                 attention_type: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
                 caption_channels: int = 3840,
                 positional_embedding_theta: float = 10000.0,
                 positional_embedding_max_pos: list[int] | None = None,
                 timestep_scale_multiplier: int = 1000,
                 use_middle_indices_grid: bool = True,
                 rope_type: LTXRopeType = LTXRopeType.SPLIT,
                 double_precision_rope: bool = True,
                 normalize_rope_positions: bool = True,
                 normalize_time_by_fps: bool = True,
                 # LTX 2.3 new params
                 apply_gated_attention: bool = False,
                 cross_attention_adaln: bool = False,
                 caption_proj_before_connector: bool = False,
                 # Legacy params
                 patch_size=(1, 1, 1),
                 text_len=1024,
                 in_dim=128,
                 dim=4096,
                 ffn_dim=16384,
                 text_dim=3840,
                 out_dim=128,
                 num_heads=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 ):

        super().__init__()

        attention_type = AttentionFunction.FLASH_ATTENTION_3
        self.model_type = LTX23ModelType.VideoOnly
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # LTX2 specific params
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.use_middle_indices_grid = use_middle_indices_grid
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        print("[!]using rope type: ",rope_type)
        self.double_precision_rope = double_precision_rope
        self.normalize_rope_positions = normalize_rope_positions
        self.normalize_time_by_fps = normalize_time_by_fps
        self.yarn_rope = False
        self.yarn_max_train_seconds = 20.0

        # LTX 2.3 new params
        self.apply_gated_attention = apply_gated_attention
        self.cross_attention_adaln = cross_attention_adaln

        if positional_embedding_max_pos is None:
            positional_embedding_max_pos = [20, 2048, 2048]
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim

        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)

        # AdaLN: coefficient depends on cross_attention_adaln
        _adaln_coeff = adaln_embedding_coefficient(cross_attention_adaln)
        self.adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=_adaln_coeff)

        # Prompt AdaLN for cross-attention (NEW in 2.3)
        self.prompt_adaln_single = (
            AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2) if cross_attention_adaln else None
        )

        # 22B models (caption_proj_before_connector=True): projection is in text encoder, not transformer
        self.caption_proj_before_connector = caption_proj_before_connector
        if not caption_proj_before_connector:
            self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=self.inner_dim)
        else:
            self.caption_projection = None
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

        # Transformer blocks
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=cross_attention_adaln,
            )
            if self.model_type.is_video_enabled()
            else None
        )

        self.blocks = torch.nn.ModuleList(
            [
                LTX23AttentionBlock(
                    idx=idx,
                    video=video_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_function=attention_type,
                )
                for idx in range(num_layers)
            ]
        )

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def set_gradient_checkpointing(self, enable: bool) -> None:
        self.gradient_checkpointing = enable

    def precompute_cam_text_embeddings(self, encode_fn, device, dtype=torch.bfloat16):
        """Pre-encode 81 action texts using text encoder and store as raw buffer.

        Raw embeddings (text_dim space) are stored; projection through
        caption_projection happens in forward() after concatenation with caption.

        Args:
            encode_fn: text encoder function, takes List[str] → List[Tensor[S, text_dim]]
            device: target device
            dtype: target dtype
        """
        from wan.modules.model import ACTION_TEXT_TABLE
        action_texts = ACTION_TEXT_TABLE  # 81 strings
        print(f"[LTX23-CamText] Encoding {len(action_texts)} action texts...")

        # Encode all 81 texts
        raw_embs = encode_fn(action_texts)  # List of Tensor[S_i, text_dim]

        # 直接存 List[Tensor]，按 index 取
        self._cam_text_raw = [e.to(device=device, dtype=dtype) for e in raw_embs]
        self._use_cam_text_cross_attn = True

        print(f"[LTX23-CamText] Stored {len(self._cam_text_raw)} raw embeddings, "
              f"token_lens={[e.shape[0] for e in self._cam_text_raw[:5]]}..., "
              f"text_dim={self._cam_text_raw[0].shape[1]}")
        for i in [0, 1, 3, 40]:
            if i < len(action_texts):
                print(f"[LTX23-CamText]   label={i}: \"{action_texts[i]}\" → {self._cam_text_raw[i].shape[0]} tokens")

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(self, x, t, context, seq_len,
                action_labels: Optional[torch.Tensor] = None,
                perturbations: "BatchedPerturbationConfig | None" = None,
                **kwargs):
        return self._forward(x, t, context, seq_len,
                             action_labels=action_labels,
                             cond_latent_frames=kwargs.get('cond_latent_frames', 0),
                             perturbations=perturbations,
                             **{k: v for k, v in kwargs.items() if k != 'cond_latent_frames'})

    def _forward(
        self, x, t, context, seq_len,
        fps: float = 25.0,
        action_labels: Optional[torch.Tensor] = None,
        cond_latent_frames: int = 0,
        perturbations: "BatchedPerturbationConfig | None" = None,
        **kwargs,
    ):
        from ltx23.modules.patchifier import VideoLatentPatchifier, VideoLatentShape, SpatioTemporalScaleFactors, get_pixel_coords

        device = self.patchify_proj.weight.device
        dtype = self.patchify_proj.weight.dtype

        # x: List[[C, F, H, W]] -> [B, C, F, H, W]
        if isinstance(x, list):
            x_stacked = torch.stack([u for u in x])
        else:
            x_stacked = x

        batch_size, channels, num_frames, height, width = x_stacked.shape
        # Patchify
        patchifier = VideoLatentPatchifier(patch_size=1)
        target_shape = VideoLatentShape(
            batch=batch_size, channels=channels, frames=num_frames,
            height=height, width=width
        )
        latent = patchifier.patchify(x_stacked).to(dtype)
        num_tokens = latent.shape[1]

        # Sigma computation
        t_float = t.float()
        if t_float.max() > 1.0:
            sigma_val = t_float / 1000.0
        else:
            sigma_val = t_float
        timesteps = sigma_val.unsqueeze(-1).unsqueeze(-1).expand(-1, num_tokens, 1).clone().to(dtype)

        # i2v/v2v conditioning
        if cond_latent_frames > 0:
            tokens_per_frame = height * width
            cond_tokens = min(cond_latent_frames * tokens_per_frame, num_tokens)
            timesteps[:, :cond_tokens, :] = 0.0

        if not hasattr(self, '_ts_print_count'):
            self._ts_print_count = 0
        # [torch.compile] print 已注释，避免 dynamo trace 报错
        # Positions with FPS normalization
        scale_factors = SpatioTemporalScaleFactors(time=8, width=32, height=32)
        latent_coords = patchifier.get_patch_grid_bounds(output_shape=target_shape, device=device)
        positions = get_pixel_coords(
            latent_coords=latent_coords, scale_factors=scale_factors, causal_fix=True
        ).float()

        if getattr(self, 'normalize_time_by_fps', True):
            positions[:, 0, ...] = positions[:, 0, ...] / fps
        positions = positions.to(dtype)

        # Context
        if isinstance(context, list):
            max_len = max(c.shape[0] for c in context)
            context_padded = []
            for c in context:
                if c.shape[0] < max_len:
                    pad = torch.zeros(max_len - c.shape[0], c.shape[1], device=c.device, dtype=c.dtype)
                    c = torch.cat([c, pad], dim=0)
                context_padded.append(c)
            context_tensor = torch.stack(context_padded).to(dtype)
        else:
            context_tensor = context.to(dtype)

        # Patch embedding
        hidden = self.patchify_proj(latent)

        # Time embeddings (optimized: only compute unique sigmas)
        tokens_per_frame = height * width
        cond_tokens = cond_latent_frames * tokens_per_frame if cond_latent_frames > 0 else 0

        if cond_latent_frames > 0 and cond_tokens < num_tokens:
            sigma_gen = timesteps[0, -1, 0:1]
            unique_sigmas = torch.stack([torch.zeros_like(sigma_gen), sigma_gen])
            unique_scaled = unique_sigmas * self.timestep_scale_multiplier
            unique_emb, unique_embedded = self.adaln_single(
                unique_scaled.squeeze(-1), hidden_dtype=hidden.dtype,
            )

            token_indices = torch.ones(num_tokens, dtype=torch.long, device=device)
            token_indices[:cond_tokens] = 0
            timestep_emb = unique_emb[token_indices].unsqueeze(0)
            embedded_timestep = unique_embedded[token_indices].unsqueeze(0)
        else:
            single_sigma = timesteps[0, 0, 0:1]
            single_scaled = single_sigma * self.timestep_scale_multiplier
            single_emb, single_embedded = self.adaln_single(
                single_scaled, hidden_dtype=hidden.dtype,
            )
            timestep_emb = single_emb.unsqueeze(0).expand(batch_size, num_tokens, -1)
            embedded_timestep = single_embedded.unsqueeze(0).expand(batch_size, num_tokens, -1)

        # Prompt timestep for cross-attention AdaLN (NEW in 2.3)
        # prompt_timestep should be (B, 1, 2*dim) to broadcast over context tokens
        prompt_timestep = None
        if self.prompt_adaln_single is not None:
            if cond_latent_frames > 0 and cond_tokens < num_tokens:
                unique_prompt_emb, _ = self.prompt_adaln_single(
                    unique_scaled.squeeze(-1), hidden_dtype=hidden.dtype,
                )
                # For i2v with different sigmas per token, use sigma=0 for cond and actual sigma for gen
                # Take a single representative (the gen token sigma) for prompt timestep
                prompt_timestep = unique_prompt_emb[1:2].unsqueeze(0).expand(batch_size, 1, -1)
            else:
                single_prompt_emb, _ = self.prompt_adaln_single(
                    single_scaled, hidden_dtype=hidden.dtype,
                )
                prompt_timestep = single_prompt_emb.unsqueeze(0).expand(batch_size, 1, -1)

        # Caption projection (None for 22B where projection is in text encoder)
        if self.caption_projection is not None:
            context_proj = self.caption_projection(context_tensor)
            context_proj = context_proj.view(batch_size, -1, hidden.shape[-1])
        else:
            context_proj = context_tensor.view(batch_size, -1, hidden.shape[-1])

        # --- Per-frame camera text cross-attention ---
        per_frame_context = None
        _tokens_per_frame = None
        _num_latent_frames = None

        _use_cam_text = (getattr(self, '_use_cam_text_cross_attn', False)
                         and action_labels is not None
                         and hasattr(self, '_cam_text_raw'))

        if _use_cam_text:
            T = num_frames  # latent frames
            H_p = height    # latent height
            W_p = width     # latent width
            _tokens_per_frame = H_p * W_p
            _num_latent_frames = T

            al = action_labels
            if al.shape[1] != T:
                al = F.interpolate(
                    al.float().unsqueeze(1), size=T, mode='nearest'
                ).squeeze(1).long()
            assert al.min() >= 0 and al.max() < len(self._cam_text_raw), \
                f"action_labels out of range: min={al.min().item()}, max={al.max().item()}, num_actions={len(self._cam_text_raw)}"

            text_dim_raw = self._cam_text_raw[0].shape[1]  # text_dim (e.g. 4096)
            S_target = context_tensor.shape[1]  # max context length

            # 确保 context_tensor 是 3D [B, S, D]（validation 时可能多一个 batch dim）
            if context_tensor.dim() == 4:
                context_tensor = context_tensor.squeeze(1)
            elif context_tensor.dim() == 2:
                context_tensor = context_tensor.unsqueeze(0)

            # 逐帧构建 [cam_text, caption, zero_pad] 在 raw text_dim 空间
            per_frame_raw = []
            for b in range(batch_size):
                cap_b = context_tensor[b]  # [S_cap, text_dim]
                for t_idx in range(T):
                    aid = al[b, t_idx].item()
                    cam = self._cam_text_raw[aid]  # [S_i, text_dim]
                    combined = torch.cat([cam, cap_b], dim=0)
                    if combined.shape[0] >= S_target:
                        per_frame_raw.append(combined[:S_target])
                    else:
                        pad = combined.new_zeros(S_target - combined.shape[0], text_dim_raw)
                        per_frame_raw.append(torch.cat([combined, pad], dim=0))

            per_frame_raw = torch.stack(per_frame_raw)  # [B*T, S_target, text_dim]

            # cam + caption 一起过 caption_projection 投影到 hidden_dim
            if self.caption_projection is not None:
                per_frame_context = self.caption_projection(per_frame_raw)
                per_frame_context = per_frame_context.view(batch_size * T, -1, hidden.shape[-1])
            else:
                per_frame_context = per_frame_raw.view(batch_size * T, -1, hidden.shape[-1])

        # Dummy forward for prompt_adaln_single FSDP sync
        if self.prompt_adaln_single is not None and prompt_timestep is None:
            _dummy_t = torch.zeros(1, device=device, dtype=dtype)
            _ = self.prompt_adaln_single(_dummy_t, hidden_dtype=dtype)

        # Positional embeddings (RoPE)
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        freqs = precompute_freqs_cis(
            indices_grid=positions,
            dim=self.inner_dim,
            out_dtype=latent.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
            normalize_positions=self.normalize_rope_positions,
        )

        # Context Parallel
        _cp_orig_seq_len = None
        _cp_debug = is_cp_enabled() and not hasattr(self, '_cp_debug_done')
        if is_cp_enabled():
            cp_size = get_cp_world_size()
            if _cp_debug:
                _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                print(f"[CP-Debug][rank={_rank}] 进入 CP scatter, hidden={hidden.shape}, seq={hidden.shape[1]}", flush=True)
            hidden, _cp_orig_seq_len = pad_to_cp_divisible(hidden, dim=1)
            timestep_emb, _ = pad_to_cp_divisible(timestep_emb, dim=1)
            if prompt_timestep is not None:
                prompt_timestep, _ = pad_to_cp_divisible(prompt_timestep, dim=1)
            freqs_cos, _ = pad_to_cp_divisible(freqs[0], dim=2)
            freqs_sin, _ = pad_to_cp_divisible(freqs[1], dim=2)
            freqs = (freqs_cos, freqs_sin)
            if _cp_debug:
                print(f"[CP-Debug][rank={_rank}] pad 完成, hidden={hidden.shape}", flush=True)
            hidden = scatter_sequence(hidden, dim=1)
            if _cp_debug:
                print(f"[CP-Debug][rank={_rank}] scatter hidden 完成, hidden={hidden.shape}", flush=True)
            timestep_emb = scatter_sequence(timestep_emb, dim=1)
            if prompt_timestep is not None:
                prompt_timestep = scatter_sequence(prompt_timestep, dim=1)
            freqs = (scatter_sequence(freqs[0], dim=2), scatter_sequence(freqs[1], dim=2))
            if _cp_debug:
                print(f"[CP-Debug][rank={_rank}] CP scatter 全部完成, hidden={hidden.shape}", flush=True)

        # Transformer blocks
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for _blk_idx, block in enumerate(self.blocks):
            if _cp_debug and _blk_idx < 3:
                print(f"[CP-Debug][rank={_rank}] block {_blk_idx} 开始, hidden={hidden.shape}", flush=True)
            block_kwargs = {
                "timesteps": timestep_emb,
                "freqs": freqs,
                "context": context_proj,
                "context_mask": None,
                "perturbations": perturbations,
                "prompt_timestep": prompt_timestep,
                "self_attention_mask": None,
                "per_frame_context": per_frame_context,
                "tokens_per_frame": _tokens_per_frame,
                "num_latent_frames": _num_latent_frames,
            }
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden,
                    **block_kwargs,
                    use_reentrant=False,
                )
            else:
                hidden = block(x=hidden, **block_kwargs)
            if _cp_debug and _blk_idx < 3:
                print(f"[CP-Debug][rank={_rank}] block {_blk_idx} 完成", flush=True)

        if _cp_debug:
            print(f"[CP-Debug][rank={_rank}] 所有 blocks 完成, 开始 gather", flush=True)

        # Gather from context parallel
        if is_cp_enabled():
            hidden = gather_for_loss(hidden, dim=1)
            if _cp_orig_seq_len is not None:
                hidden = unpad_from_cp(hidden, _cp_orig_seq_len, dim=1)
            if _cp_debug:
                print(f"[CP-Debug][rank={_rank}] gather 完成, hidden={hidden.shape}", flush=True)
                self._cp_debug_done = True  # 只打印第一次 forward

        # Output processing
        out = self._process_output(self.scale_shift_table, self.norm_out, self.proj_out, hidden, embedded_timestep)

        # Unpatchify
        out_bcfhw = patchifier.unpatchify(out, target_shape)

        return out_bcfhw

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.model_type.is_video_enabled():
            nn.init.xavier_uniform_(self.patchify_proj.weight)
            if self.caption_projection is not None:
                for m in self.caption_projection.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, std=.02)
            for m in self.adaln_single.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=.02)
            nn.init.zeros_(self.proj_out.weight)


# =============================================================================
# Utility Functions
# =============================================================================

def to_denoised(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float | torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype)
    return (sample.to(calc_dtype) - velocity.to(calc_dtype) * sigma).to(sample.dtype)


class X0Model(nn.Module):
    def __init__(self, velocity_model: LTX23Model):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(self, x, t, context, seq_len, **kwargs) -> torch.Tensor:
        velocity = self.velocity_model(x, t, context, seq_len, **kwargs)
        sigma = t.float() / 1000.0
        while sigma.dim() < velocity.dim():
            sigma = sigma.unsqueeze(-1)
        if isinstance(x, list):
            x_stacked = torch.stack(x)
        else:
            x_stacked = x
        x0 = x_stacked - sigma * velocity
        return x0
