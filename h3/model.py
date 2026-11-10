# SPDX-License-Identifier: Apache-2.0
"""Native H3 with camera PRoPE, full bidirectional attention, and no HAND branch.

Derived from SolarWM's stage0.5_v7 frozen adapter. Camera controls are passed
through each block's forward, so activation recomputation retains its own poses.
"""
from dataclasses import dataclass

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM, MiniMaxH3AttnProcessor,
    MiniMaxH3Transformer3DModel, MiniMaxH3TransformerOutput, _apply_rotary_emb,
)
from diffusers.utils import apply_lora_scale

from .diffusers_compat import patch_minimax_h3_parameter_dtype
from .torch_prope import prope_qkv

get_dtype = patch_minimax_h3_parameter_dtype()


@dataclass(frozen=True)
class CameraControl:
    indices: torch.Tensor
    views: torch.Tensor
    intrinsics: torch.Tensor


class CameraAttnProcessor(MiniMaxH3AttnProcessor):
    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if not isinstance(attention_mask, CameraControl):
            return super().__call__(attn, hidden_states, rotary_emb, attention_mask)
        control = attention_mask
        if rotary_emb is None:
            raise ValueError("Camera PRoPE requires native MM-RoPE")
        if attn.fused_projections:
            q, k, v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            q, k, v = (getattr(attn, f"to_{name}")(hidden_states) for name in ("q", "k", "v"))
        q, k, v = (x.unflatten(-1, (attn.heads, -1)) for x in (q, k, v))
        q = _apply_rotary_emb(attn.norm_q(q), *rotary_emb)
        k = _apply_rotary_emb(attn.norm_k(k), *rotary_emb)
        indices = control.indices
        selected = [x.index_select(1, indices).transpose(1, 2) for x in (q, k, v)]
        pq, pk, pv, restore = prope_qkv(
            *selected, viewmats=control.views.to(q.dtype), Ks=control.intrinsics.to(q.dtype))
        q, k, v = (x.index_copy(1, indices, p.transpose(1, 2))
                   for x, p in zip((q, k, v), (pq, pk, pv)))
        attended = dispatch_attention_fn(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
            backend=getattr(self, "_attention_backend", None),
            parallel_config=getattr(self, "_parallel_config", None))
        restored = restore(attended.index_select(1, indices).transpose(1, 2)).transpose(1, 2)
        attended = attended.index_copy(1, indices, restored).flatten(2, 3).type_as(q)
        return attn.to_out[1](attn.to_out[0](attended))


class BiWMH3Transformer(MiniMaxH3Transformer3DModel):
    def install_camera_processors(self):
        for block in self.transformer_blocks:
            block.attn.set_processor(CameraAttnProcessor())

    @classmethod
    def from_config(cls, config=None, return_unused_kwargs=False, **kwargs):
        result = super().from_config(config, return_unused_kwargs=return_unused_kwargs, **kwargs)
        model = result[0] if return_unused_kwargs else result
        model.install_camera_processors()
        return result

    @classmethod
    def strict_from_pretrained(cls, path, **kwargs):
        model, info = cls.from_pretrained(path, output_loading_info=True, **kwargs)
        errors = {key: info[key] for key in (
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
        if errors:
            raise RuntimeError(f"H3 checkpoint did not strict-load: {errors}")
        mismatches = []
        for name, parameter in model.named_parameters():
            fp32 = name.startswith(("proj_in.", "proj_out.", "audio_proj_in.",
                                    "audio_proj_out.", "time_embedder."))
            expected = torch.float32 if fp32 else torch.bfloat16
            if parameter.dtype != expected:
                mismatches.append(f"{name}: {parameter.dtype}, expected {expected}")
        if mismatches:
            raise RuntimeError(f"H3 mixed-precision checkpoint contract differs: {mismatches[:8]}")
        return model

    @apply_lora_scale("attention_kwargs")
    def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states,
                timestep, timestep_indices, token_tags, position_ids, video_indices,
                audio_indices, text_indices, attention_kwargs=None, return_dict=True,
                *, cam_viewmats, cam_K, camera_frame_ids):
        del attention_kwargs
        sequence = len(position_ids)
        if token_tags.shape != (sequence,) or timestep_indices.shape != (sequence,):
            raise ValueError("Packed H3 row indices differ")
        if len(audio_indices) or audio_hidden_states.shape[1]:
            raise ValueError("BiWM H3 Stage 1 accepts video only")
        if camera_frame_ids.shape != video_indices.shape:
            raise ValueError("Camera mapping must cover anchor and target video rows")
        if cam_viewmats.ndim != 4 or cam_viewmats.shape[-2:] != (4, 4):
            raise ValueError("Camera views must be [B,T,4,4]")
        if cam_K.shape != (*cam_viewmats.shape[:2], 3, 3):
            raise ValueError("Camera K must be [B,T,3,3]")
        control = CameraControl(video_indices,
                                cam_viewmats.index_select(1, camera_frame_ids),
                                cam_K.index_select(1, camera_frame_ids))
        rotary = self.rope(position_ids)
        video = self.proj_in(hidden_states.to(get_dtype(self.proj_in)))
        text = self.token_refiner(self.context_embedder(
            encoder_hidden_states.to(get_dtype(self.context_embedder))))
        packed = text.new_zeros(text.shape[0], sequence, text.shape[-1])
        packed = packed.index_copy(1, text_indices, text)
        packed = packed.index_copy(1, video_indices, video.to(text.dtype))
        temb = self.time_embedder(self.time_proj(timestep).to(get_dtype(self.time_embedder)))
        adaln = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                packed = self._gradient_checkpointing_func(block, packed, temb, adaln, rotary, control)
            else:
                packed = block(packed, temb, adaln, rotary, control)
        packed = self.norm_out(packed, temb, timestep_indices).to(get_dtype(self.proj_out))
        video = self.proj_out(packed).index_select(1, video_indices)
        audio = video.new_empty(video.shape[0], 0, audio_hidden_states.shape[-1])
        if not return_dict:
            return video, audio
        return MiniMaxH3TransformerOutput(sample=video, audio_sample=audio)
