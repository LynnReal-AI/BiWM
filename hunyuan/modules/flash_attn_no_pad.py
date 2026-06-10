# Vendored from HunyuanVideo-1.5 (Tencent Hunyuan Community License); imports rewritten to hunyuan.* local deps.
# rsh 2026-06-10 07:52:18 comment cleanup (no logic change)

import math
import os
import torch
from einops import rearrange
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask

DEFAULT_FLEX_BLOCK_SIZE = 128

# compile first
flex_attention = torch.compile(
    flex_attention, mode="default")

def prepare_blockwise_causal_attn_mask(
            device: torch.device | str, num_frames: int = 21,
            frame_seqlen: int = 880, causal_mask=None
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """

        total_length = num_frames * frame_seqlen

        def attention_mask(b, h, q_idx, kv_idx):
            return causal_mask[q_idx, kv_idx]

def _dense_block_rows_to_ordered(dense_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a dense block mask [num_q_blocks, num_kv_blocks] to ordered BlockMask metadata."""
    dense_mask = dense_mask.to(dtype=torch.int32)
    num_blocks = dense_mask.sum(dim=-1).to(dtype=torch.int32, memory_format=torch.contiguous_format)
    indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
    indices = indices.to(dtype=torch.int32, memory_format=torch.contiguous_format)
    return num_blocks, indices


def _get_tf_flex_block_size() -> int:
    raw = os.environ.get("HYVIDEO_TF_FLEX_BLOCK_SIZE")
    if raw is None:
        return DEFAULT_FLEX_BLOCK_SIZE
    block_size = int(raw)
    if block_size <= 0:
        raise ValueError(
            f"HYVIDEO_TF_FLEX_BLOCK_SIZE must be positive, got {block_size}"
        )
    return block_size


def _iter_teacher_forcing_q_segments(
    q_start: int,
    q_end: int,
    text_seq_length: int,
    clean_start: int,
    noisy_start: int,
    total_length: int,
    attention_block_size: int,
):
    """Yield query sub-segments where the teacher-forcing attention rule is constant.
     Q layout:
        [0 ......... text_seq_length) [clean_start ......... noisy_start)
        [noisy_start ......... total_length) [padding...)
    
    """
    text_end = min(q_end, clean_start)
    if q_start < clean_start and text_end > q_start:
        yield ("text", q_start, text_end, text_seq_length)

    clean_q_start = max(q_start, clean_start)
    clean_q_end = min(q_end, noisy_start)
    if clean_q_start < clean_q_end:
        block_idx = (clean_q_start - clean_start) // attention_block_size
        seg_start = clean_q_start
        while seg_start < clean_q_end:
            clean_block_end = min(
                clean_start + (block_idx + 1) * attention_block_size,
                noisy_start,
            )
            seg_end = min(clean_q_end, clean_block_end)
            yield ("clean", seg_start, seg_end, clean_block_end)
            seg_start = seg_end
            block_idx += 1

    noisy_q_start = max(q_start, noisy_start)
    noisy_q_end = min(q_end, total_length)
    if noisy_q_start < noisy_q_end:
        block_idx = (noisy_q_start - noisy_start) // attention_block_size
        seg_start = noisy_q_start
        while seg_start < noisy_q_end:
            noisy_block_start = noisy_start + block_idx * attention_block_size
            noisy_block_end = min(noisy_block_start + attention_block_size, total_length)
            seg_end = min(noisy_q_end, noisy_block_end)
            clean_context_end = clean_start + block_idx * attention_block_size
            yield (
                "noisy",
                seg_start,
                seg_end,
                clean_context_end,
                noisy_block_start,
                noisy_block_end,
            )
            seg_start = seg_end
            block_idx += 1

    pad_start = max(q_start, total_length)
    if pad_start < q_end:
        yield ("pad", pad_start, q_end, text_seq_length)


def _vec_interval_full_any(kv_starts, kv_ends, allowed_start, allowed_end):
    """Vectorized interval check across all kv blocks. Returns (full, any) bool tensors."""
    if allowed_end <= allowed_start:
        z = torch.zeros(kv_starts.shape[0], dtype=torch.bool)
        return z, z.clone()
    full = (allowed_start <= kv_starts) & (kv_ends <= allowed_end)
    any_hit = (kv_starts < allowed_end) & (allowed_start < kv_ends)
    return full, any_hit


def _build_teacher_forcing_block_mask_direct(
    device,
    num_frames: int,
    frame_seqlen: int,
    text_seq_length: int,
    num_frame_per_block: int = 4,
    block_size: int = DEFAULT_FLEX_BLOCK_SIZE,
) -> BlockMask:
    """Build BlockMask metadata directly in block space to avoid create_block_mask's dense O(S^2) mask."""
    vision_length = num_frames * frame_seqlen
    total_length = text_seq_length + vision_length * 2
    padded_total = math.ceil(total_length / 128) * 128
    noisy_start = text_seq_length + vision_length
    attention_block_size = frame_seqlen * num_frame_per_block

    # Build the mask_mod closure (same logic as prepare_teacher_forcing_mask)
    _tsl = text_seq_length
    _ns = noisy_start
    _ne = noisy_start + vision_length
    _abs = attention_block_size

    def attention_mask(b, h, q_idx, kv_idx):
        text_mask = kv_idx < _tsl
        q_clean_block_end = _tsl + ((q_idx - _tsl) // _abs + 1) * _abs
        clean_mask = (
            (q_idx >= _tsl) & (q_idx < _ns)
            & (kv_idx >= _tsl) & (kv_idx < q_clean_block_end)
            & (kv_idx < _ns)
        )
        noisy_block_idx = (q_idx - _ns) // _abs
        noisy_to_clean = (
            (q_idx >= _ns) & (q_idx < _ne)
            & (kv_idx >= _tsl)
            & (kv_idx < _tsl + noisy_block_idx * _abs)
        )
        noisy_block_start = _ns + noisy_block_idx * _abs
        noisy_to_noisy = (
            (q_idx >= _ns) & (q_idx < _ne)
            & (kv_idx >= noisy_block_start)
            & (kv_idx < noisy_block_start + _abs)
            & (kv_idx < _ne)
        )
        return (q_idx == kv_idx) | text_mask | clean_mask | noisy_to_clean | noisy_to_noisy

    num_blocks = math.ceil(padded_total / block_size)

    # Precompute all kv block boundaries once
    kv_starts = torch.arange(num_blocks, dtype=torch.int64) * block_size
    kv_ends = torch.clamp(kv_starts + block_size, max=padded_total)

    full_mask = torch.zeros((num_blocks, num_blocks), dtype=torch.bool)
    partial_mask = torch.zeros((num_blocks, num_blocks), dtype=torch.bool)

    for q_bi in range(num_blocks):
        q_start = q_bi * block_size
        q_end = min(q_start + block_size, padded_total)

        segments = list(
            _iter_teacher_forcing_q_segments(
                q_start=q_start,
                q_end=q_end,
                text_seq_length=text_seq_length,
                clean_start=text_seq_length,
                noisy_start=noisy_start,
                total_length=total_length,
                attention_block_size=attention_block_size,
            )
        )
        if not segments:
            continue

        # Vectorized: check all kv blocks at once per segment
        row_full = torch.ones(num_blocks, dtype=torch.bool)
        row_any = torch.zeros(num_blocks, dtype=torch.bool)

        for seg in segments:
            kind = seg[0]
            if kind == "text":
                sf, sa = _vec_interval_full_any(kv_starts, kv_ends, 0, seg[3])
            elif kind == "clean":
                sf, sa = _vec_interval_full_any(kv_starts, kv_ends, 0, seg[3])
            elif kind == "noisy":
                cf, ca = _vec_interval_full_any(kv_starts, kv_ends, 0, seg[3])
                nf, na = _vec_interval_full_any(kv_starts, kv_ends, seg[4], seg[5])
                sf = cf | nf
                sa = ca | na
            elif kind == "pad":
                tf, ta = _vec_interval_full_any(kv_starts, kv_ends, 0, seg[3])
                eye_any = (kv_starts < seg[2]) & (seg[1] < kv_ends)
                sf = tf
                sa = ta | eye_any
            else:
                raise ValueError(f"Unknown segment type: {kind}")

            row_full &= sf
            row_any |= sa

        full_mask[q_bi] = row_full
        partial_mask[q_bi] = row_any & ~row_full

    partial_num_blocks, partial_indices = _dense_block_rows_to_ordered(partial_mask)
    if full_mask.any():
        full_num_blocks, full_indices = _dense_block_rows_to_ordered(full_mask)
        full_num_blocks = full_num_blocks.unsqueeze(0).unsqueeze(0).to(device)
        full_indices = full_indices.unsqueeze(0).unsqueeze(0).to(device)
    else:
        full_num_blocks = None
        full_indices = None

    partial_num_blocks = partial_num_blocks.unsqueeze(0).unsqueeze(0).to(device)
    partial_indices = partial_indices.unsqueeze(0).unsqueeze(0).to(device)

    return BlockMask.from_kv_blocks(
        partial_num_blocks,
        partial_indices,
        full_num_blocks,
        full_indices,
        BLOCK_SIZE=block_size,
        mask_mod=attention_mask,
        seq_lengths=(padded_total, padded_total),
    )


def prepare_teacher_forcing_mask(
    device,
    num_frames: int,
    frame_seqlen: int,
    text_seq_length: int,
    num_frame_per_block: int = 4,
) -> BlockMask:
    """Build a BlockMask for teacher forcing training (memory-efficient, block-space)."""
    block_size = _get_tf_flex_block_size()
    block_mask = _build_teacher_forcing_block_mask_direct(
        device=device,
        num_frames=num_frames,
        frame_seqlen=frame_seqlen,
        text_seq_length=text_seq_length,
        num_frame_per_block=num_frame_per_block,
        block_size=block_size,
    )

    vision_length = num_frames * frame_seqlen
    total_length = text_seq_length + vision_length * 2
    clean_start = text_seq_length
    noisy_start = text_seq_length + vision_length
    attention_block_size = frame_seqlen * num_frame_per_block

    return block_mask



def prepare_teacher_forcing_mask_old(
    device,
    num_frames: int,
    frame_seqlen: int,
    text_seq_length: int,
    num_frame_per_block: int = 4,
) -> BlockMask:
    """Build a BlockMask for teacher forcing training (dense create_block_mask version)."""
    vision_length = num_frames * frame_seqlen
    total_length = text_seq_length + vision_length * 2  # text + clean + noisy

    # pad to multiple of 128 for flex_attention
    padded_total = math.ceil(total_length / 128) * 128
    padded_length = padded_total - total_length

    clean_start = text_seq_length
    noisy_start = text_seq_length + vision_length
    attention_block_size = frame_seqlen * num_frame_per_block

    # For clean tokens: block causal end indices
    context_ends = torch.zeros(padded_total, device=device, dtype=torch.long)
    # For noisy tokens: two intervals [context_start, context_end] and [noise_start, noise_end]
    noise_context_starts = torch.zeros(padded_total, device=device, dtype=torch.long)
    noise_context_ends = torch.zeros(padded_total, device=device, dtype=torch.long)
    noise_noise_starts = torch.zeros(padded_total, device=device, dtype=torch.long)
    noise_noise_ends = torch.zeros(padded_total, device=device, dtype=torch.long)

    # Clean blocks: block causal
    for start in range(clean_start, noisy_start, attention_block_size):
        end = min(start + attention_block_size, noisy_start)
        context_ends[start:end] = end

    # Noisy blocks
    for block_idx, ns in enumerate(range(noisy_start, noisy_start + vision_length, attention_block_size)):
        ne = min(ns + attention_block_size, noisy_start + vision_length)
        # attend to own noisy block
        noise_noise_starts[ns:ne] = ns
        noise_noise_ends[ns:ne] = ne
        # attend to preceding clean blocks (0..block_idx-1)
        noise_context_starts[ns:ne] = clean_start
        noise_context_ends[ns:ne] = clean_start + block_idx * attention_block_size

    _text_seq_length = text_seq_length
    _clean_start = clean_start
    _noisy_start = noisy_start

    def attention_mask(b, h, q_idx, kv_idx):
        # All tokens can attend to text
        text_mask = kv_idx < _text_seq_length

        # Clean-to-Clean: block causal
        clean_mask = (
            (q_idx >= _clean_start) & (q_idx < _noisy_start)
            & (kv_idx >= _clean_start) & (kv_idx < context_ends[q_idx])
        )

        # Noisy-to-Clean: attend to preceding clean blocks
        noisy_to_clean = (
            (q_idx >= _noisy_start)
            & (kv_idx >= noise_context_starts[q_idx])
            & (kv_idx < noise_context_ends[q_idx])
        )

        # Noisy-to-Noisy: attend to own block
        noisy_to_noisy = (
            (q_idx >= _noisy_start)
            & (kv_idx >= noise_noise_starts[q_idx])
            & (kv_idx < noise_noise_ends[q_idx])
        )

        eye_mask = q_idx == kv_idx
        return eye_mask | text_mask | clean_mask | noisy_to_clean | noisy_to_noisy

    block_mask = create_block_mask(
        attention_mask, B=None, H=None,
        Q_LEN=padded_total, KV_LEN=padded_total,
        _compile=False, device=device,
    )

    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        _visualize_block_mask(block_mask, total_length, text_seq_length, clean_start, noisy_start, attention_block_size)

    return block_mask


def _visualize_block_mask(block_mask, total_length, text_seq_length, clean_start, noisy_start, attention_block_size):
    """Visualize token-level attention mask by reconstructing from attention rules."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        # Build token-level mask using the attention_mask function
        _tsl = text_seq_length
        _cs = clean_start
        _ns = noisy_start
        _ne = total_length
        _abs = attention_block_size

        def attention_mask(q_idx, kv_idx):
            text_mask = kv_idx < _tsl
            q_clean_block_end = _cs + ((q_idx - _cs) // _abs + 1) * _abs
            clean_mask = ((q_idx >= _cs) & (q_idx < _ns) & (kv_idx >= _cs) &
                         (kv_idx < q_clean_block_end) & (kv_idx < _ns))
            noisy_block_idx = (q_idx - _ns) // _abs
            noisy_to_clean = ((q_idx >= _ns) & (q_idx < _ne) & (kv_idx >= _cs) &
                             (kv_idx < _cs + noisy_block_idx * _abs))
            noisy_block_start = _ns + noisy_block_idx * _abs
            noisy_block_end = noisy_block_start + _abs
            noisy_to_noisy = ((q_idx >= _ns) & (q_idx < _ne) & (kv_idx >= noisy_block_start) &
                             (kv_idx < noisy_block_end) & (kv_idx < _ne))
            eye_mask = q_idx == kv_idx
            return eye_mask | text_mask | clean_mask | noisy_to_clean | noisy_to_noisy

        # Generate token-level mask (subsample if too large)
        max_vis_size = 2048
        if total_length <= max_vis_size:
            q_indices = np.arange(total_length)
            kv_indices = np.arange(total_length)
            stride = 1
        else:
            stride = (total_length + max_vis_size - 1) // max_vis_size
            q_indices = np.arange(0, total_length, stride)
            kv_indices = np.arange(0, total_length, stride)

        dense = np.zeros((len(q_indices), len(kv_indices)), dtype=np.float32)
        for i, q_idx in enumerate(q_indices):
            for j, kv_idx in enumerate(kv_indices):
                dense[i, j] = float(attention_mask(q_idx, kv_idx))

        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(dense, cmap='gray_r', vmin=0, vmax=1, origin='upper', aspect='auto', interpolation='nearest')

        title = f'Token-level Attention Mask (black=attend, white=masked)'
        if stride > 1:
            title += f'\nSubsampled: stride={stride}, showing {len(q_indices)}x{len(kv_indices)} of {total_length}x{total_length}'
        ax.set_title(title)
        ax.set_xlabel('KV token index')
        ax.set_ylabel('Q token index')

        # Add region boundaries
        if stride == 1:
            ax.axvline(text_seq_length - 0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(noisy_start - 0.5, color='blue', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(text_seq_length - 0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(noisy_start - 0.5, color='blue', linestyle='--', linewidth=1, alpha=0.5)
        else:
            ax.axvline(text_seq_length / stride - 0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(noisy_start / stride - 0.5, color='blue', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(text_seq_length / stride - 0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(noisy_start / stride - 0.5, color='blue', linestyle='--', linewidth=1, alpha=0.5)

        save_path = 'attention_mask_vis.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"[BlockMask] Saved token-level visualization to {save_path}")
    except Exception as e:
        print(f"[BlockMask] Visualization failed: {e}")


# add new function for computing causal attention with flex attention
def flex_attn_no_pad(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False
):
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    batch_size = qkv.shape[0]    # qkv shape: [B, total_length, 3, num_head, D]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(
        x, key_padding_mask
    )

    # ------------------------- for chunk-wise causal attention mask -------------------------
    # get the video sequence
    latent_seq_length = 1560      # set for hunyuanvideo 1.5, which is for 480 * 832 resolution
    chunk_seq_length = 1560 * 4
    text_seq_length = max_s % chunk_seq_length    # including txt, byt5, sigclip vision token
    latent_num = (max_s - text_seq_length) // latent_seq_length
    chunk_num = (max_s - text_seq_length) // chunk_seq_length
    causal_mask = torch.zeros((max_s, max_s), device=qkv.device)

    causal_mask[:text_seq_length, :text_seq_length] = 1  # no attention for the rest
    for i in range(chunk_num):
        start_i = text_seq_length + i * chunk_seq_length
        end_i = min(start_i + chunk_seq_length, max_s)
        for j in range(i + 1):
            start_j = text_seq_length + j * chunk_seq_length
            end_j = min(start_j + chunk_seq_length, max_s)
            # full attention within chunk i for j == i, causal for j < i
            causal_mask[start_i:end_i, start_j:end_j] = 1
    causal_mask = causal_mask.to(torch.bool)  # Force bool dtype

    block_mask = prepare_blockwise_causal_attn_mask(
                device=qkv.device,
                num_frames=latent_num,
                frame_seqlen=latent_seq_length,
                causal_mask=causal_mask
            )

    q,k,v = qkv.chunk(3, dim=2)
    q = q.squeeze(2).transpose(1, 2)
    k = k.squeeze(2).transpose(1, 2)
    v = v.squeeze(2).transpose(1, 2)
    output_unpad = flex_attention(
                q, k, v, block_mask=block_mask
            )   # output_unpad: [B, num_head, L, D]

    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


# add new function for computing causal attention with flex attention
def flash_attn_no_pad(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False
):
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(
        x, key_padding_mask
    )

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


def flash_attn_no_pad_v3(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False
):
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3

    if flash_attn_varlen_func_v3 is None:
        raise ImportError("FlashAttention V3 backend not available")
    
    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    query, key, value = qkv.unbind(dim=2)
    
    query_unpad, indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
        rearrange(query, "b s h d -> b s (h d)"), key_padding_mask
    )
    key_unpad, _, cu_seqlens_k, _, _ = unpad_input(
        rearrange(key, "b s h d -> b s (h d)"), key_padding_mask
    )
    value_unpad, _, _, _, _ = unpad_input(
        rearrange(value, "b s h d -> b s (h d)"), key_padding_mask
    )
    
    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=nheads)
    key_unpad = rearrange(key_unpad, "nnz (h d) -> nnz h d", h=nheads)
    value_unpad = rearrange(value_unpad, "nnz (h d) -> nnz h d", h=nheads)
    
    output_unpad = flash_attn_varlen_func_v3(
        query_unpad, key_unpad, value_unpad,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_q, 
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic
    )
    
    output = rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
        "b s (h d) -> b s h d", h=nheads
    )
    return output
