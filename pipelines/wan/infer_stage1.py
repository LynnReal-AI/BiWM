# >>> BiWM Stage 1 — inference (multi-step CFG teacher sampling) <<<
"""
Stage1 (stage-1 cam-text model) multi-step CFG t2v inference — single video output (not combined), with new keycap/joystick overlay.
  - No distillation, non-autoregressive: use data_teacher_sample to generate the whole clip at once with num_steps (default 50) CFG steps (same as the training validation teacher).
  - Camera: action_frames (single-axis segments) or action_label (video_real constant combination); per-latent-frame cam-text cross-attn.
  - Each case is saved twice: <out>.mp4 (no joystick) + <out>_control.mp4 (new keycap/joystick).
  - The interface matches infer_stage2 (--cases_json/--case_index), and can use the 8-GPU data-parallel dispatch of infer_dmd_multi.sh
    (ENTRY=pipelines/wan/infer_stage1.py). The passed-through autoregressive params (duration/chunk/max_chunks/sink) are ignored by this script.

Single case: python pipelines/wan/infer_stage1.py --generator_ckpt <stage1 ckpt dir> --wan_base <Wan2.2-TI2V-5B> \
        --prompt "..." --action_frames "w-8,right-12,s-6"
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import safetensors.torch

from pipelines.wan.wan_common import (
    init_wan_transformer, init_wan_vae, init_wan_text_encoder,
    wan_vae_reconstruct, store_video,
    WAN_VAE_TIME_STRIDE, WAN_LATENT_CHANNEL_COUNT, WAN_NEG_PROMPT_TEXT,
    WAN_VAE_SPACE_STRIDE, WAN21_LATENT_CHANNEL_COUNT, WAN21_VAE_SPACE_STRIDE,
    resolve_wan_version,
)
from pipelines.wan.dmd_core import data_teacher_sample
from pipelines.wan.infer_stage2 import make_action_labels
from pipelines.common.control_overlay import superimpose_control_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator_ckpt", required=True, help="stage1 ckpt dir or .safetensors")
    ap.add_argument("--wan_base", required=True, help="Wan base model dir (architecture+VAE+text encoder); Wan2.2-TI2V-5B or Wan2.1-T2V-1.3B")
    ap.add_argument("--wan_version", type=str, default="auto", choices=["auto", "2.1", "2.2"],
                    help="Wan backbone version: 'auto' detects from --wan_base path, or force '2.1'(1.3B)/'2.2'(5B).")
    ap.add_argument("--cases_json", default=None)
    ap.add_argument("--case_index", type=int, default=0)
    ap.add_argument("--prompt", default="A cinematic scene, smooth camera movement, highly detailed.")
    ap.add_argument("--action_frames", default=None, help="single-axis segment pose (e.g. 'w-8,right-12,s-6'); empty=random")
    ap.add_argument("--action_label", type=int, default=None, help="constant 81-class combined action (for video_real)")
    ap.add_argument("--num_steps", type=int, default=50, help="number of CFG sampling steps")
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--num_frames", type=int, default=77, help="number of pixel frames (77≈3.2s@24fps → 20 latent)")
    ap.add_argument("--num_height", type=int, default=480)
    ap.add_argument("--num_width", type=int, default=832)
    ap.add_argument("--sigma_shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--output", default="./outputs/stage1_infer.mp4")
    ap.add_argument("--no_control", action="store_true", help="save only the original version without joystick")
    ap.add_argument("--control", action="store_true", help="[for infer_dmd_multi pass-through compatibility, now both versions are saved by default]")
    # ★ For infer_dmd_multi.sh pass-through compatibility: autoregressive params (ignored by this script)
    for ig in ["duration_sec", "chunk_size", "max_chunks", "sink_chunks", "biwm_video_dir"]:
        ap.add_argument(f"--{ig}", default=None)
    args = ap.parse_args()
    args.dmd_timestep_shift = args.sigma_shift     # data_teacher_sample reads it to get shift

    # ---- Multiple cases: case override ----
    if args.cases_json:
        import json
        _doc = json.load(open(args.cases_json, "r", encoding="utf-8"))
        _cases = _doc["cases"] if isinstance(_doc, dict) else _doc
        assert 0 <= args.case_index < len(_cases), f"case_index out of range (total {len(_cases)})"
        c = _cases[args.case_index]
        args.prompt = c["prompt"]
        args.action_frames = c.get("action_frames", args.action_frames)
        args.action_label = c.get("action_label", args.action_label)
        args.seed = int(c.get("seed", args.seed))
        cid = c.get("id", f"case{args.case_index:02d}")
        if args.output == "./outputs/stage1_infer.mp4":
            args.output = f"./outputs/stage1_infer/{args.case_index:02d}_{cid}_seed{args.seed}.mp4"
        print(f"[Stage1][case {args.case_index}] id={cid} seed={args.seed} "
              f"pose={args.action_frames} action_label={args.action_label}", flush=True)

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    version = resolve_wan_version(args)
    _SP = WAN21_VAE_SPACE_STRIDE if version == '2.1' else WAN_VAE_SPACE_STRIDE
    C = WAN21_LATENT_CHANNEL_COUNT if version == '2.1' else WAN_LATENT_CHANNEL_COUNT
    Hl, Wl = args.num_height // _SP, args.num_width // _SP
    T_lat = (args.num_frames - 1) // WAN_VAE_TIME_STRIDE + 1

    # ---- Load: text encoder → model(+stage1 weights + cam-text) → vae ----
    print(f"[Stage1] loading text encoder / model / vae ...", flush=True)
    text_encoder, tokenizer, encode_fn = init_wan_text_encoder(args.wan_base, device, dtype)
    model = init_wan_transformer(args.wan_base, device, dtype, model_type="ti2v", version=version)
    ckpt = args.generator_ckpt
    if os.path.isdir(ckpt):
        ckpt = os.path.join(ckpt, "diffusion_pytorch_model.safetensors")
    sd = safetensors.torch.load_file(ckpt, device="cpu")
    miss, unexp = model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"[Stage1] stage1 <- {ckpt}: {len(sd)} tensors, missing={len(miss)} unexpected={len(unexp)}")
    model.precompute_cam_text_embeddings(encode_fn, device, dtype)
    _, vae_decoder = init_wan_vae(args.wan_base, device, dtype, version=version)

    # ---- text embedding (CFG: cond=caption+cam-text, uncond=neg caption) ----
    caption_emb = encode_fn([args.prompt])[0].to(device=device, dtype=dtype).unsqueeze(0)
    neg_emb = encode_fn([WAN_NEG_PROMPT_TEXT])[0].to(device=device, dtype=dtype).unsqueeze(0)
    full_labels = make_action_labels(T_lat, args.action_frames, args.seed,
                                      action_label=args.action_label).to(device)
    print(f"[Stage1] prompt='{args.prompt[:60]}...' T_lat={T_lat} num_steps={args.num_steps} cfg={args.cfg}",
          flush=True)

    # ---- 50-step CFG t2v full-clip generation ----
    t0 = time.time()
    x0 = data_teacher_sample(model, caption_emb, neg_emb, full_labels, (C, Hl, Wl), T_lat,
                             device, dtype, args, num_steps=args.num_steps, cfg=args.cfg)
    frames = wan_vae_reconstruct(vae_decoder, x0[0], device)
    print(f"[Stage1] decoded {len(frames)} frames, generation time {time.time()-t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    store_video(frames, args.output, fps=int(round(args.fps)))
    print(f"[Stage1] ✓ saved (no joystick): {args.output}")
    if not args.no_control:
        frames_js = superimpose_control_video(frames, full_labels[0, :T_lat],
                                           vae_temporal_stride=WAN_VAE_TIME_STRIDE)
        jp = os.path.splitext(args.output)[0] + "_control.mp4"
        store_video(frames_js, jp, fps=int(round(args.fps)))
        print(f"[Stage1] ✓ saved (new keycap/joystick): {jp}")


if __name__ == "__main__":
    main()
