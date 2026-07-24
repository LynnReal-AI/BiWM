#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2026-06-10 01:35:00
"""DMD 4-step autoregressive inference (generator trained by stage2_dmd.sh).

Per-chunk autoregressive generation consistent with the training no-HE path: each chunk = chunk_size
latent frames, full 4-step denoising (no CFG); sliding window of max_chunks (+ optional sink chunks);
history = already-generated clean latents (sigma=0) as cond frames; per-frame discrete camera action_labels.
Supports t2v (chunk0 from pure noise) and i2v (VAE-encode first frame as initial history).
"""
import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import safetensors.torch

# Same loader / decode / save / constants as in training
from pipelines.wan.wan_common import (
    init_wan_transformer, init_wan_vae, init_wan_text_encoder,
    wan_vae_reconstruct, store_video,
    WAN_VAE_TIME_STRIDE, WAN_LATENT_CHANNEL_COUNT, WAN_NEG_PROMPT_TEXT,
    WAN_VAE_SPACE_STRIDE, WAN21_LATENT_CHANNEL_COUNT, WAN21_VAE_SPACE_STRIDE,
    resolve_wan_version,
)
# Same velocity wrapper + flow x0 as in training
from pipelines.wan.dmd_core import _dit_velocity_field, velocity_into_x0, WAN_PATCH_H, WAN_PATCH_W
# pose parsing + token→label mapping (same set as the training dataset)
from pipelines.dataset.biwm_camera_text_dataset import pose_text_to_motion_labels, TOKEN_LABEL_LOOKUP


# single-token discrete action set (biWM dataset): static + 4 translations + 4 rotations
_DEFAULT_LABELS = [0,                      # static
                TOKEN_LABEL_LOOKUP["w"], TOKEN_LABEL_LOOKUP["s"],            # forward/backward
                TOKEN_LABEL_LOOKUP["a"], TOKEN_LABEL_LOOKUP["d"],            # left/right translation
                TOKEN_LABEL_LOOKUP["up"], TOKEN_LABEL_LOOKUP["down"],        # pitch up/down
                TOKEN_LABEL_LOOKUP["left"], TOKEN_LABEL_LOOKUP["right"]]     # turn left/right


def make_action_labels(num_latent_frames, action_frames=None, seed=0, action_label=None):
    """Return per-latent-frame discrete labels (0~80) of shape [1, num_latent_frames].
    action_label given → a single constant [combined] action over the whole segment (used by video_real, init frame 0, rest=label);
    otherwise if action_frames is given, parse it (e.g. "w-8, right-12, s-6"); if neither, construct randomly."""
    if action_label is not None:
        seq = [0] + [int(action_label)] * max(0, num_latent_frames - 1)
        labels = torch.tensor(seq[:num_latent_frames], dtype=torch.long)
    elif action_frames:
        labels = pose_text_to_motion_labels(action_frames, num_latent_frames)  # [L]
    else:
        rng = random.Random(seed)
        seq = [0]                              # frame 0 fixed to static (init)
        while len(seq) < num_latent_frames:
            lab = rng.choice(_DEFAULT_LABELS)
            dur = rng.randint(4, 16)           # each action lasts 4~16 latent frames
            seq.extend([lab] * dur)
        labels = torch.tensor(seq[:num_latent_frames], dtype=torch.long)
    return labels.unsqueeze(0)                 # [1, L]


# single-chunk 4-step rollout (no CFG, equivalent to student_rollout train_mode=False)
@torch.no_grad()
def unroll_chunk(model, prefix, caption_emb, al_window, sched,
                  chunk_size, C, H, W, tpf, device, dtype, batch=1):
    """Generate chunk_size latent frames for one chunk (supports B parallel batches, for throughput testing).
      prefix: [B,C,n_hist,H,W] clean history frames (sigma=0 condition frames), None=no history (t2v chunk0).
      al_window: [B, n_hist+chunk_size] per-frame action_labels for the whole window.
    Returns x0: [B,C,chunk_size,H,W] clean latent."""
    n_steps = len(sched) - 1
    n_hist = prefix.shape[2] if prefix is not None else 0

    def _vel(x_cur, s):
        if prefix is not None:
            x_in = torch.cat([prefix, x_cur], dim=2)            # [1,C,n_hist+K,H,W]
            seq_len = (n_hist + chunk_size) * tpf
            v_full = _dit_velocity_field(model, x_in, s, caption_emb, seq_len,
                                           cond_latent_frames=n_hist, action_labels=al_window)
            return v_full[:, :, n_hist:, :, :]                  # take only the current chunk
        else:
            seq_len = chunk_size * tpf
            target_t = torch.arange(0, chunk_size, device=device, dtype=torch.float32)
            return _dit_velocity_field(model, x_cur, s, caption_emb, seq_len,
                                         target_t_indices=target_t, action_labels=al_window)

    x_t = torch.randn(batch, C, chunk_size, H, W, device=device, dtype=dtype)
    x0 = None
    for i in range(n_steps):
        s_cur = sched[i]
        v = _vel(x_t, s_cur)
        x0 = velocity_into_x0(v, x_t, s_cur).to(dtype)
        if i < n_steps - 1:                                     # re-add noise to the next sigma level
            s_nxt = sched[i + 1]
            x_t = ((1 - s_nxt) * x0 + s_nxt * torch.randn_like(x0)).to(dtype)
    return x0


# i2v: read the first frame of a dataset video → VAE latent
def embed_first_frame(video_path, vae_encoder, num_height, num_width, device, dtype):
    """Read the first frame of a video, preprocess ([-1,1], resize to the training resolution), VAE encode → [1,48,1,H_lat,W_lat].
    Returns (lat, first_frame_rgb): first_frame_rgb is the actual model condition frame (resized RGB uint8 [H,W,3]), for saving."""
    from decord import VideoReader, cpu
    from PIL import Image
    import torchvision.transforms as T
    vr = VideoReader(video_path, ctx=cpu(0))
    img = Image.fromarray(vr[0].asnumpy())                      # first frame
    t = T.ToTensor()(img)                                       # [C,H,W] in [0,1]
    t = F.interpolate(t.unsqueeze(0), size=(num_height, num_width),
                      mode="bicubic", align_corners=False, antialias=True).squeeze(0)
    # the condition frame the model actually sees (after resize, [0,1]→uint8 RGB), for saving
    first_frame_rgb = (t.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    t = t.sub_(0.5).div_(0.5)                                   # -> [-1,1]  (consistent with training _process_frame)
    video = t.unsqueeze(1).to(device=device, dtype=dtype)       # [C, T=1, H, W]
    lat = vae_encoder.encode([video])[0]                        # [48, 1, H_lat, W_lat]
    return lat.unsqueeze(0).to(device=device, dtype=dtype), first_frame_rgb  # ([1,48,1,H_lat,W_lat], [H,W,3])


# VAE decode note: the Wan2.2 VAE is a causal decoder; decode() iterates per latent frame + holds feat_cache
#   across frames, so any length is decoded in one pass. Do not do segment-wise decode (restarts the causal cache
#   -> boundary frame-count errors + seams + joystick drift). pixel->latent: pf=0->latent0; pf>=1->(pf-1)//stride+1.
def main():
    ap = argparse.ArgumentParser("DMD 4-step autoregressive inference")
    # model / weights
    ap.add_argument("--generator_ckpt", required=True,
                    help="trained generator ckpt directory (containing diffusion_pytorch_model.safetensors) or a .safetensors file")
    ap.add_argument("--wan_base", required=True, help="Wan base model directory (architecture + VAE + text encoder); Wan2.2-TI2V-5B or Wan2.1-T2V-1.3B")
    ap.add_argument("--wan_version", type=str, default="auto", choices=["auto", "2.1", "2.2"],
                    help="Wan backbone version: 'auto' detects from --wan_base path, or force '2.1'(1.3B)/'2.2'(5B).")
    # mode
    ap.add_argument("--mode", choices=["t2v", "i2v"], default="t2v")
    ap.add_argument("--prompt", default="A cinematic scene, smooth camera movement, highly detailed, photorealistic.")
    ap.add_argument("--i2v_video", default=None,
                    help="path to the i2v first-frame source video; if empty, automatically takes the first gen.mp4 under dataset/videos")
    ap.add_argument("--biwm_video_dir",
                    default="./dataset/videos")
    # pose / duration
    ap.add_argument("--duration_sec", type=float, default=60.0, help="total autoregressive generation duration (seconds)")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--action_frames", default=None, help="specify camera pose (e.g. 'w-8,right-12,s-6'); empty=randomly constructed")
    ap.add_argument("--action_label", type=int, default=None,
                    help="a single constant 81-class combined action over the whole segment (used by video_real; mutually exclusive with action_frames, takes priority)")
    # rollout configuration (aligned with stage2_dmd.sh)
    ap.add_argument("--chunk_size", type=int, default=4, help="latent frames per chunk (=training dmd_block_K)")
    ap.add_argument("--max_chunks", type=int, default=5,
                    help="max number of chunks K in the context window, discards the earliest chunk when exceeded (=training dmd_num_blocks upper limit)")
    ap.add_argument("--sink_chunks", type=int, default=1,
                    help="number of sink chunks: when the sliding window discards, these first few chunks are permanently kept (attention sink, StreamingLLM style); "
                         "0=disabled (pure sliding window, drop earliest). Default 1=the first chunk is never discarded")
    ap.add_argument("--dmd_sigmas", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25],
                    help="4-step starting sigma (before shift; trailing 0.0 implied)")
    ap.add_argument("--sigma_shift", type=float, default=5.0)
    # resolution
    ap.add_argument("--num_height", type=int, default=480)
    ap.add_argument("--num_width", type=int, default=832)
    # output
    ap.add_argument("--output", default="./outputs/dmd_infer.mp4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=1,
                    help="generate B videos in parallel")
    ap.add_argument("--decode_seg_latent", type=int, default=0,
                    help="[deprecated/ignored] the Wan VAE is causal+cache, a single decode suffices; the buggy segment-wise decode has been removed")
    ap.add_argument("--control", action="store_true",
                    help="[for backward compatibility with old scripts, both versions are now saved by default] joystick visualization")
    ap.add_argument("--no_control", action="store_true",
                    help="only save the joystick-free original video, no extra joystick version (both versions saved by default)")
    # multi-sample testing: take the case_index-th case from JSON to override prompt/action_frames/seed/mode (for 8-GPU distribution)
    ap.add_argument("--cases_json", default=None,
                    help="test cases JSON (list or {cases:[...]}, each item contains prompt/action_frames, optionally mode/seed/id)")
    ap.add_argument("--case_index", type=int, default=0, help="which case in cases_json to use")
    args = ap.parse_args()

    # multi-sample: override prompt/action_frames/seed/mode with the case from JSON (output auto-named)
    if args.cases_json:
        import json
        with open(args.cases_json, "r", encoding="utf-8") as f:
            _doc = json.load(f)
        _cases = _doc["cases"] if isinstance(_doc, dict) else _doc
        assert 0 <= args.case_index < len(_cases),\
            f"case_index={args.case_index} out of range (total {len(_cases)} cases)"
        _case = _cases[args.case_index]
        args.prompt = _case["prompt"]
        args.action_frames = _case.get("action_frames", args.action_frames)
        args.action_label = _case.get("action_label", args.action_label)   # video_real constant combined action
        args.mode = _case.get("mode", args.mode)
        args.seed = int(_case.get("seed", args.seed))
        args.i2v_video = _case.get("i2v_video", args.i2v_video)   # i2v first-frame source (per-case)
        _cid = _case.get("id", f"case{args.case_index:02d}")
        if args.output == "./outputs/dmd_infer.mp4":     # output not explicitly given → name by case id
            args.output = f"./outputs/dmd_infer/{args.case_index:02d}_{_cid}_seed{args.seed}.mp4"
        print(f"[Infer][case {args.case_index}] id={_cid} seed={args.seed} mode={args.mode} "
              f"i2v_video={args.i2v_video}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # latent dimensions (version-dependent: 2.2-5B = 16x/48ch, 2.1-1.3B = 8x/16ch)
    version = resolve_wan_version(args)
    _SP = WAN21_VAE_SPACE_STRIDE if version == '2.1' else WAN_VAE_SPACE_STRIDE
    C = WAN21_LATENT_CHANNEL_COUNT if version == '2.1' else WAN_LATENT_CHANNEL_COUNT
    Hl = args.num_height // _SP
    Wl = args.num_width // _SP
    tpf = (Hl // WAN_PATCH_H) * (Wl // WAN_PATCH_W)
    chunk_size = args.chunk_size
    max_ctx_frames = (args.max_chunks - 1) * chunk_size        # history frame upper limit

    # 4-step schedule (DMD_SIGMAS + timestep_shift, consistent with training)
    sched = list(args.dmd_sigmas) + [0.0]
    shift = float(args.sigma_shift)
    if shift and shift != 1.0:
        sched = [(shift * s / (1 + (shift - 1) * s)) for s in sched]
    print(f"[Infer] 4-step sched (shift={shift}) = {[round(s,4) for s in sched]}")

    # load: text encoder → model(+cam_text) → vae
    print(f"[Infer] loading text encoder / model / vae ...")
    text_encoder, tokenizer, encode_fn = init_wan_text_encoder(args.wan_base, device, dtype)
    model = init_wan_transformer(args.wan_base, device, dtype, model_type="ti2v", version=version)
    ckpt_file = (os.path.join(args.generator_ckpt, "diffusion_pytorch_model.safetensors")
                 if os.path.isdir(args.generator_ckpt) else args.generator_ckpt)
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"generator weights not found: {ckpt_file}")
    sd = safetensors.torch.load_file(ckpt_file, device="cpu")
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[Infer] generator <- {ckpt_file}: loaded {len(sd)} tensors, missing={len(miss)}, unexpected={len(unexp)}")
    model = model.to(device=device, dtype=dtype).eval()
    model.precompute_cam_text_embeddings(encode_fn, device, dtype)    # cam-text (bf16), consistent with training
    vae_encoder, vae_decoder = init_wan_vae(args.wan_base, device, dtype, version=version)

    # caption embedding (no CFG, the generator needs no neg); expand to B when batch>1
    _B = max(1, int(args.batch_size))
    with torch.no_grad():
        caption_emb = encode_fn([args.prompt])[0].to(device=device, dtype=dtype).unsqueeze(0)
        if _B > 1:
            caption_emb = caption_emb.expand(_B, -1, -1).contiguous()   # [B,S,dim]
    print(f"[Infer] prompt='{args.prompt[:60]}...' cap_emb={list(caption_emb.shape)} batch={_B}")

    # target latent frame count / chunk count
    target_pixel = int(round(args.duration_sec * args.fps))
    target_latent = max(chunk_size, (target_pixel - 1) // WAN_VAE_TIME_STRIDE + 1)
    seed_frames = 1 if args.mode == "i2v" else 0
    num_chunks = math.ceil(max(0, target_latent - seed_frames) / chunk_size)
    total_latent = seed_frames + num_chunks * chunk_size

    # pose / action_labels (covering the whole segment); expand to B when batch>1 (same pose)
    full_labels = make_action_labels(total_latent + chunk_size, args.action_frames, args.seed,
                                      action_label=getattr(args, 'action_label', None)).to(device)
    if _B > 1:
        full_labels = full_labels.expand(_B, -1).contiguous()          # [B, L]
    print(f"[Infer] mode={args.mode} duration={args.duration_sec}s fps={args.fps} "
          f"→ target_latent≈{target_latent}, num_chunks={num_chunks}, total_latent={total_latent}, "
          f"window≤{args.max_chunks}chunk({args.max_chunks*chunk_size}latent), "
          f"sink={args.sink_chunks}chunk({max(0,args.sink_chunks)*chunk_size}latent)")

    # i2v first-frame latent
    clean_lat = None
    if args.mode == "i2v":
        v_path = args.i2v_video
        if not v_path:
            cand = sorted(d for d in os.listdir(args.biwm_video_dir)
                          if os.path.isdir(os.path.join(args.biwm_video_dir, d)))
            assert cand, f"no video directory under {args.biwm_video_dir}"
            v_path = os.path.join(args.biwm_video_dir, cand[0], "gen.mp4")
        print(f"[Infer] i2v first-frame source: {v_path}")
        clean_lat, first_frame_rgb = embed_first_frame(
            v_path, vae_encoder, args.num_height, args.num_width, device, dtype)
        # save the actual model condition first frame next to the output for comparison
        from PIL import Image
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        _ff_path = os.path.splitext(args.output)[0] + "_firstframe.png"
        Image.fromarray(first_frame_rgb).save(_ff_path)
        print(f"[Infer] i2v condition first frame saved: {_ff_path}")
        if _B > 1:
            clean_lat = clean_lat.expand(_B, -1, -1, -1, -1).contiguous()   # [B,C,1,H,W]

    # per-chunk autoregressive 4-step. sink chunk: when the sliding window discards, the first sink_n frames are
    # permanently kept (attention sink, StreamingLLM style). Window not full -> all history; window full ->
    # [first sink_n] + [most recent (max_ctx-sink_n)], middle discarded. prefix/al_window gather the same absolute indices.
    sink_n_cap = max(0, args.sink_chunks) * chunk_size      # sink: number of leading frames permanently kept
    for c in range(num_chunks):
        start = clean_lat.shape[2] if clean_lat is not None else 0
        if clean_lat is not None and max_ctx_frames > 0:
            total_hist = clean_lat.shape[2]
            if total_hist <= max_ctx_frames:
                hist_idx = torch.arange(0, total_hist, device=device)        # window not full, keep all
            else:
                sink_n = min(sink_n_cap, max_ctx_frames)                     # number of sink frames (capped at window)
                recent_budget = max_ctx_frames - sink_n                      # budget left for recent frames
                recent_start = total_hist - recent_budget
                hist_idx = torch.cat([
                    torch.arange(0, sink_n, device=device),                  # sink (permanent)
                    torch.arange(recent_start, total_hist, device=device),   # recent (sliding)
                ])
            prefix = clean_lat[:, :, hist_idx, :, :].contiguous()
            n_hist = prefix.shape[2]
            cur_idx = torch.arange(start, start + chunk_size, device=device)
            al_window = full_labels[:, torch.cat([hist_idx, cur_idx])]       # strictly aligned with the prefix frames
        else:
            prefix = None
            n_hist = 0
            al_window = full_labels[:, start: start + chunk_size]
        x0 = unroll_chunk(model, prefix, caption_emb, al_window, sched,
                           chunk_size, C, Hl, Wl, tpf, device, dtype, batch=_B)
        clean_lat = x0 if clean_lat is None else torch.cat([clean_lat, x0], dim=2)
        if c % 5 == 0 or c == num_chunks - 1:
            print(f"[Infer] chunk {c+1}/{num_chunks} done | total_latent={clean_lat.shape[2]} "
                  f"| n_hist={n_hist} | x0 std={x0.float().std().item():.3f}", flush=True)

    # VAE decode (single causal pass, feat_cache throughout) + save
    print(f"[Infer] VAE decode (single causal pass, {clean_lat.shape[2]} latent, batch={_B} decode only item 0) ...")
    frames = wan_vae_reconstruct(vae_decoder, clean_lat[0], device)   # batch>1 decodes only item 0
    print(f"[Infer] decoded {len(frames)} frames ({len(frames)/args.fps:.1f}s @ {args.fps}fps)")

    # save two videos: (1) joystick-free original -> args.output; (2) joystick-overlaid -> <output>_control.mp4 (unless --no_control)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    store_video(frames, args.output, fps=int(round(args.fps)))
    print(f"[Infer] ✓ saved (joystick-free): {args.output}")
    if not getattr(args, 'no_control', False):
        from pipelines.common.control_overlay import superimpose_control_video
        # pixel→latent uses the default formula (pf-1)//stride+1, consistent with decode
        frames_js = superimpose_control_video(frames, full_labels[0, :clean_lat.shape[2]],
                                           vae_temporal_stride=WAN_VAE_TIME_STRIDE)
        _js_path = os.path.splitext(args.output)[0] + "_control.mp4"
        store_video(frames_js, _js_path, fps=int(round(args.fps)))
        print(f"[Infer] ✓ saved (with joystick): {_js_path}")

if __name__ == "__main__":
    main()
