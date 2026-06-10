# Vendored from HunyuanVideo-1.5 (Tencent Hunyuan Community License); imports rewritten to hunyuan.* local deps.
# rsh 2026-06-10 07:52:18 comment cleanup (no logic change)

import einops
import torch
from typing import Optional
try:
    from loguru import logger
except ImportError:  # slimmed version has no loguru, fall back to stdlib logging
    import logging
    logger = logging.getLogger('hunyuan')
import numpy as np
import torch.nn.functional as F

from ..distributed.parallel_states import get_parallel_state
from ..distributed.communication_op import (
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_to_all_4D,
)
from .flash_attn_no_pad import (
    flash_attn_no_pad,
    flash_attn_no_pad_v3,
    flex_attn_no_pad
)
from .commons import maybe_fallback_attn_mode

try:
    from torch.nn.attention.flex_attention import flex_attention
    flex_attention = torch.compile(flex_attention)
    torch._dynamo.config.cache_size_limit = 192
    torch._dynamo.config.accumulated_cache_size_limit = 192
    flex_mask_cache = {}
except ImportError:
    flex_attention = None
    flex_mask_cache = {}


from .ssta_attention import ssta_3d_attention
from .infer_state import get_infer_state


@torch.compiler.disable
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    drop_rate: float = 0.0,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    attn_mode: str = "flash",
) -> torch.Tensor:
    """Compute attention using flash_attn_no_pad or torch scaled_dot_product_attention."""
    attn_mode = maybe_fallback_attn_mode(attn_mode)

    if attn_mode == "torch":
        # transpose q,k,v dim to fit scaled_dot_product_attention
        query = q.transpose(1, 2)  # B * H * L * D
        key = k.transpose(1, 2)  # B * H * L * D
        value = v.transpose(1, 2)  # B * H * L * D

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool and attn_mask.dtype in [
                torch.int64,
                torch.int32,
            ]:
                assert (
                    attn_mask.max() <= 1 and attn_mask.min() >= 0
                ), f"attention mask must be (0,1)."
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(query.dtype)
                raise NotImplementedError(
                    f"Float attention mask is not implemented for torch attention."
                )
            attn_mask1 = einops.rearrange(attn_mask, "b l -> b 1 l 1")
            attn_mask2 = einops.rearrange(attn_mask1, "b 1 l 1 -> b 1 1 l")
            attn_mask = attn_mask1 & attn_mask2

        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=drop_rate,
            is_causal=causal,
        )

        # transpose back
        x = x.transpose(1, 2)  # B * L * H * D
        b, s, h, d = x.shape
        out = x.reshape(b, s, -1)
        return out
    else:
        # flash mode (default)
        qkv = torch.stack([q, k, v], dim=2)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()
        x = flash_attn_no_pad(
            qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=None
        )
        b, s, a, d = x.shape
        out = x.reshape(b, s, -1)
        return out


@torch.compiler.disable
def sequence_parallel_attention_txt(
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    attn_mode=None,
    text_mask=None,
    attn_param=None,
    block_idx=None,
    kv_cache=None,
    cache_txt=False,
):
    encoder_query = q
    encoder_key = k
    encoder_value = v

    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled

    if enable_sp:
        sp_size = parallel_dims.sp
        sp_rank = parallel_dims.sp_rank

    if enable_sp:

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // sp_size
            return encoder_state.narrow(dim, sp_rank * local_heads, local_heads)

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)

    encoder_query = encoder_query.transpose(1, 2)
    encoder_key = encoder_key.transpose(1, 2)
    encoder_value = encoder_value.transpose(1, 2)
    t_kv = {}
    if cache_txt:
        t_kv["k_txt"] = encoder_key
        t_kv["v_txt"] = encoder_value

    infer_state = get_infer_state()
    enable_sageattn = (
        infer_state.enable_sageattn and block_idx in infer_state.sage_blocks_range
    )
    if enable_sageattn:
        from sageattention import sageattn

        encoder_hidden_states = sageattn(
            encoder_query,
            encoder_key,
            encoder_value,
            tensor_layout="HND",
            is_causal=False,
        )
    else:
        encoder_hidden_states = F.scaled_dot_product_attention(
            encoder_query, encoder_key, encoder_value, dropout_p=0.0, is_causal=False
        )

    # transpose back
    encoder_hidden_states = encoder_hidden_states.transpose(1, 2)  # [B, S, H, D]

    if enable_sp:
        encoder_hidden_states = sequence_model_parallel_all_gather(
            encoder_hidden_states, dim=2
        ).contiguous()
        encoder_hidden_states = encoder_hidden_states.to(q.dtype)

    b, s, a, d = encoder_hidden_states.shape
    encoder_hidden_states = encoder_hidden_states.reshape(b, s, -1)

    return encoder_hidden_states, t_kv


@torch.compiler.disable
def sequence_parallel_attention_vision(
    q,
    k,
    v,
    block_idx=None,
    kv_cache=None,
    cache_vision=False,
):
    assert kv_cache is not None
    query = q
    key = k
    value = v

    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled

    if enable_sp:
        query = sequence_model_parallel_all_to_all_4D(query, scatter_dim=2, gather_dim=1)
        key = sequence_model_parallel_all_to_all_4D(key, scatter_dim=2, gather_dim=1)
        value = sequence_model_parallel_all_to_all_4D(value, scatter_dim=2, gather_dim=1)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    cache_vision_key = kv_cache[block_idx]["k_vision"]
    cache_vision_value = kv_cache[block_idx]["v_vision"]

    vision_kv = {}
    if cache_vision:
        vision_kv["k_vision"] = key
        vision_kv["v_vision"] = value

    if cache_vision_key is not None:
        key = torch.cat([cache_vision_key, key], dim=2)
        value = torch.cat([cache_vision_value, value], dim=2)

    encoder_key = kv_cache[block_idx]["k_txt"]
    encoder_value = kv_cache[block_idx]["v_txt"]

    key = torch.cat([encoder_key, key], dim=2)
    value = torch.cat([encoder_value, value], dim=2)

    infer_state = get_infer_state()
    enable_sageattn = (
        infer_state.enable_sageattn and block_idx in infer_state.sage_blocks_range
    )
    if enable_sageattn:
        from sageattention import sageattn
        hidden_states = sageattn(
            query, key, value, tensor_layout="HND", is_causal=False
        )
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )

    hidden_states = hidden_states.transpose(1, 2)

    if enable_sp:
        hidden_states = sequence_model_parallel_all_to_all_4D(
            hidden_states, scatter_dim=1, gather_dim=2
        )
        hidden_states = hidden_states.to(query.dtype)

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)

    return hidden_states, vision_kv


@torch.compiler.disable
def parallel_attention(
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    attn_mode=None,
    text_mask=None,
    attn_param=None,
    block_idx=None,
    tf_block_mask=None,
):
    return sequence_parallel_attention(
        q,
        k,
        v,
        img_q_len,
        img_kv_len,
        attn_mode,
        text_mask,
        attn_param=attn_param,
        tf_block_mask=tf_block_mask,
        block_idx=block_idx,
    )


def sequence_parallel_attention(
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    attn_mode=None,
    text_mask=None,
    attn_param=None,
    tf_block_mask=None,
    block_idx=None,
):
    assert attn_mode is not None
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v

    parallel_dims = get_parallel_state()
    sp_world_size = parallel_dims.sp if parallel_dims.sp_enabled else 1

    if sp_world_size > 1:
        sp_size = sp_world_size
        sp_rank = parallel_dims.sp_rank

        # batch_size, seq_len, attn_heads, head_dim
        query = sequence_model_parallel_all_to_all_4D(query, scatter_dim=2, gather_dim=1)
        key = sequence_model_parallel_all_to_all_4D(key, scatter_dim=2, gather_dim=1)
        value = sequence_model_parallel_all_to_all_4D(value, scatter_dim=2, gather_dim=1)

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // sp_size
            return encoder_state.narrow(
                dim, sp_rank * local_heads, local_heads
            )

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    attn_mode = maybe_fallback_attn_mode(attn_mode, get_infer_state(), block_idx)

    if attn_mode == "sageattn":
        from sageattention import sageattn

        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        hidden_states = sageattn(
            query, key, value, tensor_layout="NHD", is_causal=False
        )
    elif attn_mode == "torch":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        if text_mask is not None:
            attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        else:
            attn_mask = None

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool and attn_mask.dtype in [
                torch.int64,
                torch.int32,
            ]:
                assert (
                    attn_mask.max() <= 1 and attn_mask.min() >= 0
                ), f"attention mask must be (0,1)."
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(query.dtype)
                raise NotImplementedError(
                    f"Float attention mask is not implemented for torch attention."
                )

        # transpose q,k,v dim to fit scaled_dot_product_attention
        query = query.transpose(1, 2)  # B * Head_num * length * dim
        key = key.transpose(1, 2)  # B * Head_num * length * dim
        value = value.transpose(1, 2)  # B * Head_num * length * dim
        if attn_mask is not None:
            attn_mask1 = einops.rearrange(attn_mask, "b l -> b 1 l 1")
            attn_mask2 = einops.rearrange(attn_mask1, "b 1 l 1 -> b 1 1 l")
            attn_mask = attn_mask1 & attn_mask2
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        # transpose back
        hidden_states = hidden_states.transpose(1, 2)

    # add new attn for chunk-wise attn
    elif attn_mode == "torch_causal":  # now: we set text_mask = None
        # attention: here we concat the encoder text sequence first, then apply causal attention
        vision_seq_length = query.shape[1]
        text_seq_length = encoder_query.shape[1]
        total_seq_length = vision_seq_length + text_seq_length

        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

        # prepare causal mask for chunk-wise attention
        latent_seq_length = attn_param["thw"][-1] * attn_param["thw"][-2]
        chunk_seq_length = latent_seq_length * 4
        chunk_num = (vision_seq_length) // chunk_seq_length
        causal_mask = torch.zeros(
            (total_seq_length, total_seq_length), device=query.device
        )
        causal_mask[:, :text_seq_length] = 1  # no attention for the rest
        for i in range(chunk_num):
            start_i = text_seq_length + i * chunk_seq_length
            end_i = min(start_i + chunk_seq_length, total_seq_length)
            for j in range(i + 1):
                start_j = text_seq_length + j * chunk_seq_length
                end_j = min(start_j + chunk_seq_length, total_seq_length)
                # full attention within chunk i for j == i, causal for j < i
                causal_mask[start_i:end_i, start_j:end_j] = 1

        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) # 1, 1, S, S
        causal_mask = causal_mask.expand(query.shape[0], 1, -1, -1)
        causal_mask = causal_mask.to(torch.bool)  # Force bool dtype

        query = query.transpose(1, 2)  # B * H * L * D
        key = key.transpose(1, 2)      # B * H * L * D
        value = value.transpose(1, 2)  # B * H * L * D

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=causal_mask, dropout_p=0.0, is_causal=False)
        
        # transpose back
        hidden_states = hidden_states.transpose(1, 2)   # [B, S, H, D]

        # return back to the original order: [query, encoder_query]
        hidden_states, encoder_hidden_states = hidden_states[:, text_seq_length:, :, :], hidden_states[:, :text_seq_length, :, :]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    elif attn_mode == "flex_tf":
        # Teacher forcing: use flex_attention with pre-built BlockMask
        # block_mask is passed via attn_param
        block_mask = tf_block_mask
        assert block_mask is not None, "block_mask must be provided for flex_tf mode"

        vision_seq_length = query.shape[1]  # doubled: clean + noisy
        text_seq_length = encoder_query.shape[1]

        if sp_world_size > 1:
            B, S, H, D = query.shape
            chunk_size = S // sp_world_size  # tokens per SP rank = 2 * (L/sp)
            per_rank_half = chunk_size // 2   # L/sp

            def _interleaved_to_contiguous(x):
                """[clean_0, noisy_0, clean_1, noisy_1, ...] → [all_clean, all_noisy]"""
                B, S, H, D = x.shape
                return x.reshape(B, sp_world_size, 2, per_rank_half, H, D) \
                        .permute(0, 2, 1, 3, 4, 5) \
                        .reshape(B, S, H, D)

            query = _interleaved_to_contiguous(query)
            key = _interleaved_to_contiguous(key)
            value = _interleaved_to_contiguous(value)

        # Concatenation order: [text, clean_vision, noisy_vision]
        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

        # Pad to multiple of 128 for flex_attention
        import math
        total_len = query.shape[1]
        padded_total = math.ceil(total_len / 128) * 128
        pad_len = padded_total - total_len
        if pad_len > 0:
            query = F.pad(query, (0, 0, 0, 0, 0, pad_len))
            key = F.pad(key, (0, 0, 0, 0, 0, pad_len))
            value = F.pad(value, (0, 0, 0, 0, 0, pad_len))

        # flex_attention expects [B, H, L, D]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = flex_attention(query, key, value, block_mask=block_mask)

        if pad_len > 0:
            hidden_states = hidden_states[:, :, :total_len]
        hidden_states = hidden_states.transpose(1, 2)  # [B, L, H, D]

        # Split back to vision and text
        vision_out = hidden_states[:, text_seq_length:, :, :]
        encoder_hidden_states = hidden_states[:, :text_seq_length, :, :]

        # === FIX: reverse reorder vision tokens from contiguous back to interleaved ===
        # Before reverse all-to-all, we need [clean_0, noisy_0, clean_1, noisy_1, ...]
        # so that scatter_dim=1 correctly distributes each rank's [clean_local, noisy_local].
        if sp_world_size > 1:
            B, S, H, D = vision_out.shape
            per_rank_half = S // (2 * sp_world_size)  # L/sp

            def _contiguous_to_interleaved(x):
                """[all_clean, all_noisy] → [clean_0, noisy_0, clean_1, noisy_1, ...]"""
                B, S, H, D = x.shape
                return x.reshape(B, 2, sp_world_size, per_rank_half, H, D) \
                        .permute(0, 2, 1, 3, 4, 5) \
                        .reshape(B, S, H, D)

            vision_out = _contiguous_to_interleaved(vision_out)

        # Return to [vision, text] order
        hidden_states = torch.cat([vision_out, encoder_hidden_states], dim=1)

    elif attn_mode == "flash2":
        query = torch.cat([query, encoder_query], dim=1)   # vision token first
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)

        if text_mask is not None:
            attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        else:
            # Create all-True mask when text_mask is None
            total_length = sequence_length + encoder_sequence_length
            attn_mask = torch.ones(query.shape[0], total_length, dtype=torch.bool, device=query.device)
        hidden_states = flash_attn_no_pad(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)
    
    elif attn_mode == "flex_causal":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)
        if text_mask is not None:
            attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        else:
            total_length = sequence_length + encoder_sequence_length
            attn_mask = torch.ones(query.shape[0], total_length, dtype=torch.bool, device=query.device)
        hidden_states = flex_attn_no_pad(qkv, attn_mask, causal=True, dropout_p=0.0, softmax_scale=None)

    elif attn_mode == "flash3":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)
        if text_mask is not None:
            attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        else:
            total_length = sequence_length + encoder_sequence_length
            attn_mask = torch.ones(query.shape[0], total_length, dtype=torch.bool, device=query.device)
        hidden_states = flash_attn_no_pad_v3(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)

    elif attn_mode == "flex-block-attn":
        sparse_type = attn_param["attn_sparse_type"]  # sta/block_attn/ssta
        ssta_threshold = attn_param["ssta_threshold"]
        ssta_lambda = attn_param["ssta_lambda"]
        ssta_sampling_type = attn_param["ssta_sampling_type"]
        ssta_adaptive_pool = attn_param["ssta_adaptive_pool"]

        attn_pad_type = attn_param["attn_pad_type"]  # repeat/zero
        attn_use_text_mask = attn_param["attn_use_text_mask"]
        attn_mask_share_within_head = attn_param["attn_mask_share_within_head"]

        ssta_topk = attn_param["ssta_topk"]
        thw = attn_param["thw"]
        tile_size = attn_param["tile_size"]
        win_size = attn_param["win_size"][0].copy()

        def get_image_tile(tile_size):
            block_size = np.prod(tile_size)
            if block_size == 384:
                tile_size = (1, 16, 24)
            elif block_size == 128:
                tile_size = (1, 16, 8)
            elif block_size == 64:
                tile_size = (1, 8, 8)
            elif block_size == 16:
                tile_size = (1, 4, 4)
            else:
                raise ValueError(
                    f"Error tile_size {tile_size}, only support in [16, 64, 128, 384]"
                )
            return tile_size

        if thw[0] == 1:
            tile_size = get_image_tile(tile_size)
            win_size = [1, 1, 1]
        elif thw[0] <= 31:  # 16fps: 5 * 16 / 4 + 1 = 21; 24fps: 5 * 24 / 4 + 1 = 31
            ssta_topk = ssta_topk // 2

        # Concatenate and permute query, key, value to (B, H, S, D)
        query = torch.cat([query, encoder_query], dim=1).permute(0, 2, 1, 3)
        key = torch.cat([key, encoder_key], dim=1).permute(0, 2, 1, 3)
        value = torch.cat([value, encoder_value], dim=1).permute(0, 2, 1, 3)

        assert (
            query.shape[-1] == 128
        ), "The last dimension of query, key and value must be 128 for flex-block-attn."

        hidden_states = ssta_3d_attention(
            query,
            key,
            value,
            thw,
            topk=ssta_topk,
            tile_thw=tile_size,
            kernel_thw=win_size,
            text_len=encoder_sequence_length,
            sparse_type=sparse_type,
            threshold=ssta_threshold,
            lambda_=ssta_lambda,
            pad_type=attn_pad_type,
            text_mask=text_mask if attn_use_text_mask else None,
            sampling_type=ssta_sampling_type,
            adaptive_pool=ssta_adaptive_pool,
            mask_share_within_head=attn_mask_share_within_head,
        )
        hidden_states, sparse_ratio = hidden_states
        hidden_states = hidden_states.permute(0, 2, 1, 3)

    else:
        raise NotImplementedError(
            f'Unsupported attention mode: {attn_mode}. Only torch, flash, flash3, sageattn and flex-block-attn are supported.'
        )

    if sp_world_size > 1:
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes((sequence_length, encoder_sequence_length), dim=1)
        hidden_states = sequence_model_parallel_all_to_all_4D(hidden_states, scatter_dim=1, gather_dim=2)
        encoder_hidden_states = sequence_model_parallel_all_gather(encoder_hidden_states, dim=2).contiguous()
        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states = encoder_hidden_states.to(query.dtype)
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)

    return hidden_states
