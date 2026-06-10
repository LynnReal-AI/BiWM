"""
Use Qwen3-VL-235B to caption video_real: describe only the static scene; strictly
forbid any motion (camera/camera-movement + character actions), since camera control
is a discrete text cam-text expressed separately via action_label.

Loads a single model instance (device_map="auto", model-parallel across visible GPUs),
processes clips serially writing back to json["caption"], saving every N clips for
resumable runs (skipping already-non-empty captions). Each clip uses beginning/middle/end
3 frames to recognize the scene while the prompt forbids leaking any motion.
"""
import argparse
import json
import os

import cv2
import torch
from PIL import Image

# Prompt: describe only static content; forbid any motion / inter-frame change
INSTRUCTION_TEXT = (
    "You are shown 3 frames (the beginning, middle and end) of a short video-game clip. "
    "Use them together ONLY to recognize the scene. Write ONE concise English caption "
    "(about 50-90 words) describing ONLY the static visual content present across them: the "
    "characters/avatars and their appearance (species, clothing, design), the environment and "
    "setting, notable objects, the art style, lighting, colors and overall mood.\n"
    "ABSOLUTELY FORBIDDEN — do not describe ANY motion, action, or change of any kind:\n"
    "  (a) No camera/viewpoint motion: no camera movement, panning, zooming, tilting, rotating, "
    "tracking, 'the camera', 'the shot', 'the view', 'first-person'.\n"
    "  (b) No subject/character motion or action: do NOT say anyone is running, walking, jumping, "
    "moving, turning, falling, attacking, flying, driving, etc.\n"
    "  (c) Do NOT describe any difference or change between the 3 frames.\n"
    "Describe the scene as a single frozen photograph — only what things ARE and look like, never "
    "what they DO, how they move, or how anything changes.\n"
    "Output only the caption sentence(s), no preamble."
)


def load_three_frames(mp4_path):
    """Take beginning/middle/end 3 frames (deduplicated for short clips)."""
    cap = cv2.VideoCapture(mp4_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    ids = sorted(set([0, n // 2, max(0, n - 1)]))
    frames = []
    for fid in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, f = cap.read()
        if ok and f is not None:
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError(f"cannot read frames from {mp4_path}")
    return frames


def main():
    ap = argparse.ArgumentParser()
    base = "./dataset"
    ap.add_argument("--video_dir", default=f"{base}/video_real")
    ap.add_argument("--json", default=f"{base}/videos_real.json")
    ap.add_argument("--model", default="./ckpts/Qwen3-VL-235B")
    ap.add_argument("--max_new_tokens", type=int, default=220)
    ap.add_argument("--save_every", type=int, default=20)
    args = ap.parse_args()

    from transformers import AutoProcessor
    # 235B is a MoE; AutoModelForImageTextToText picks the right class from config
    try:
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except Exception:
        from transformers import Qwen3VLMoeForConditionalGeneration as _AutoVLM
    print(f"[qwen-caption] loading {args.model} (device_map=auto, model-parallel across visible GPUs)...", flush=True)
    model = _AutoVLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto").eval()
    proc = AutoProcessor.from_pretrained(args.model)
    print("[qwen-caption] model ready", flush=True)

    with open(args.json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # id -> directory name (prefix == id)
    id2dir = {}
    for name in os.listdir(args.video_dir):
        if os.path.isdir(os.path.join(args.video_dir, name)):
            try:
                id2dir[int(name.split("_")[0])] = name
            except ValueError:
                pass

    todo = [i for i in range(len(meta)) if not meta[i].get("caption")]
    print(f"[qwen-caption] total={len(meta)} todo={len(todo)} (skipping clips that already have a caption)", flush=True)

    def _persist():
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)

    done = 0
    for k, i in enumerate(todo):
        mp4 = os.path.join(args.video_dir, id2dir[i], "gen.mp4")
        try:
            pils = load_three_frames(mp4)
            content = [{"type": "image", "image": im} for im in pils]
            content.append({"type": "text", "text": INSTRUCTION_TEXT})
            msgs = [{"role": "user", "content": content}]
            try:
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
            except Exception:
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[text], images=pils, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
            cap = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
            meta[i]["caption"] = cap.strip().replace("\n", " ")
            done += 1
        except Exception as e:
            print(f"[qwen-caption] id={i} FAILED: {e}", flush=True)
        if (k + 1) % args.save_every == 0:
            _persist()
            print(f"[qwen-caption] {k+1}/{len(todo)} done={done} saved", flush=True)
    _persist()
    n_empty = sum(1 for m in meta if not m.get("caption"))
    print(f"[qwen-caption] DONE. newly captioned {done} clips; json still empty caption {n_empty}/{len(meta)} -> {args.json}")


if __name__ == "__main__":
    main()
