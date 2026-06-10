"""
HY1.5 data pre-encoding: precompute VAE latents + text embeds for the
video_real / videos datasets so training reads .pt directly (--data_mode preenc),
skipping per-step online VAE+MLLM encoding.

Each clip is stored as one .pt:
  latent       : [1,32,T_lat,h,w] bf16  (VAE encode × scaling_factor)
  prompt_embed : [1,L1,3584] bf16  — stage1 <camera> text, padding trimmed by mask
  prompt_mask  : [1,L1] (all 1)
  prompt_embed_capt / prompt_mask_capt : caption-only text (stage2 DMD, no camera)
  caption(str) / action_label(int) / vid_id(str) : metadata

Supports RANK/WORLD_SIZE sharding (idx % world == rank); resumable (skip existing .pt).
Usage: python -m pipelines.data_preprocess.preencode_hy15 --video_dir ... --caption_json ...
       --output_dir ... --vae_path ... --text_encoder_path ...
"""
import argparse
import os
import time

import torch

from pipelines.hy15.train_hunyuan import (init_hy15_vae, init_mllm, mllm_embed,
                                  assemble_camtext_prompt, action_into_camtext,
                                  HY15_VAE_TIME_STRIDE)
from pipelines.dataset.biwm_camera_text_dataset import BiwmCamCaptionData


def _truncate(pe, pm):
    """pe:[1,L,3584] pm:[1,L] → drop padding by mask, return [1,L_real,3584]+[1,L_real] (all 1)."""
    m = pm[0].bool()
    pe_t = pe[0][m].unsqueeze(0).contiguous()                 # [1,L_real,3584]
    pm_t = torch.ones(1, pe_t.shape[1], dtype=pm.dtype)
    return pe_t, pm_t


@torch.no_grad()
def main():
    p = argparse.ArgumentParser("HY1.5 pre-encoding (VAE latent + text embed)")
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--caption_json", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--vae_path", type=str, required=True)
    p.add_argument("--text_encoder_path", type=str, required=True)
    p.add_argument("--num_frames", type=int, default=77)
    p.add_argument("--num_height", type=int, default=480)
    p.add_argument("--num_width", type=int, default=832)
    p.add_argument("--no_caption_only", action="store_true",
                   help="do not additionally encode caption-only text (only store the stage1 <camera> text)")
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    os.makedirs(args.output_dir, exist_ok=True)

    ds = BiwmCamCaptionData(
        video_dir=args.video_dir, caption_json=args.caption_json,
        width=args.num_width, height=args.num_height, num_frames=args.num_frames,
        vae_temporal_factor=HY15_VAE_TIME_STRIDE)
    n = len(ds)
    print(f"[preenc][rank{rank}/{world}] dataset={args.video_dir} total {n} clips, output -> {args.output_dir}", flush=True)

    vae = init_hy15_vae(args.vae_path, device)
    mllm = init_mllm(args.text_encoder_path, device)
    vae_dtype = next(vae.parameters()).dtype

    my_ids = [i for i in range(n) if i % world == rank]
    done = skipped = failed = 0
    t0 = time.time()
    for cnt, idx in enumerate(my_ids):
        try:
            item = ds[idx]
            if isinstance(item, dict) and item.get("skip"):
                failed += 1
                continue
            pixel_values = item["pixel_values"]              # [T,C,H,W] in [-1,1]
            caption = item.get("caption") or ""
            vid_id = item["video_id"]
            action_labels = item["action_labels"]
            out_path = os.path.join(args.output_dir, f"{vid_id}.pt")
            if os.path.exists(out_path):                      # resume
                skipped += 1
                continue

            # VAE encode → latent [1,32,T_lat,h,w] (× scaling, same as assemble_live_batch)
            x = pixel_values.permute(1, 0, 2, 3).unsqueeze(0).to(device, dtype=vae_dtype)
            latent = (vae.encode(x).latent_dist.sample() * vae._sf).to(torch.bfloat16).cpu()

            # the clip's constant action (non-zero mode, same as assemble_live_batch)
            al = torch.as_tensor(action_labels).reshape(-1).long()
            nz = al[al > 0]
            clip_action = int(torch.bincount(nz).argmax()) if nz.numel() > 0 else 0

            # stage1 text: <camera>camera</camera> static caption
            text_cam = assemble_camtext_prompt(action_into_camtext(clip_action), caption)
            pe, pm = mllm_embed(mllm, [text_cam], device)
            pe, pm = _truncate(pe.cpu(), pm.cpu())

            rec = {
                "latent": latent, "prompt_embed": pe.to(torch.bfloat16), "prompt_mask": pm.to(torch.bfloat16),
                "caption": caption, "action_label": clip_action, "vid_id": vid_id,
            }
            # stage2 caption-only text
            if not args.no_caption_only:
                pe2, pm2 = mllm_embed(mllm, [caption if caption else ""], device)
                pe2, pm2 = _truncate(pe2.cpu(), pm2.cpu())
                rec["prompt_embed_capt"] = pe2.to(torch.bfloat16)
                rec["prompt_mask_capt"] = pm2.to(torch.bfloat16)

            tmp = out_path + ".tmp"
            torch.save(rec, tmp)
            os.replace(tmp, out_path)                          # atomic write
            done += 1
            if done % 50 == 0:
                el = time.time() - t0
                rate = done / max(el, 1e-6)
                eta = (len(my_ids) - cnt - 1) / max(rate, 1e-6) / 3600
                print(f"[preenc][rank{rank}] {cnt+1}/{len(my_ids)} done={done} skip={skipped} fail={failed} "
                      f"{rate:.2f} clip/s ETA={eta:.2f}h | last={vid_id} L={pe.shape[1]} lat={tuple(latent.shape)}",
                      flush=True)
        except Exception as e:
            failed += 1
            print(f"[preenc][rank{rank}] idx={idx} failed: {type(e).__name__}: {e}", flush=True)
            continue

    print(f"[preenc][rank{rank}] finished: done={done} skip(existing)={skipped} fail={failed} / mine={len(my_ids)} "
          f"elapsed {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
