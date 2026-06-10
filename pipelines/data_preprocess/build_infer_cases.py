"""
Generate inference test-case JSON (randomly sampled, default 20 per category) → scripts/Inference/:
  - infer_cases.json            : 20 random t2v cases from dataset/videos (caption + action_frames)
  - infer_cases_i2v.json        : same 20 random cases as i2v (first frame = the clip's gen.mp4)
  - infer_cases_video_real.json : ★ 20 random i2v cases from video_real (caption + constant action_label + first frame)
Fixed seed for reproducibility. Run from repo root: python pipelines/data_preprocess/build_infer_cases.py
"""
import argparse
import json
import os
import random


def _id_to_folder(video_dir):
    m = {}
    if not os.path.isdir(video_dir):
        return m
    for n in os.listdir(video_dir):
        if os.path.isdir(os.path.join(video_dir, n)):
            try:
                m[int(n.split("_")[0])] = n
            except ValueError:
                pass
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out_dir", default="scripts/Inference")
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    # ===== dataset/videos → t2v + i2v =====
    dv = json.load(open("dataset/videos_syn.json"))
    dv_dir = "dataset/videos"
    dv_map = _id_to_folder(dv_dir)
    dv_ids = [i for i in range(len(dv)) if i in dv_map and dv[i].get("action_frames")]
    sel = rnd.sample(dv_ids, min(a.n, len(dv_ids)))
    t2v, i2v = [], []
    for k, i in enumerate(sel):
        e, d = dv[i], dv_map[i]
        t2v.append({"id": f"dv{i:06d}", "prompt": e["caption"], "action_frames": e["action_frames"],
                    "mode": "t2v", "seed": 1000 + k})
        i2v.append({"id": f"dv{i:06d}", "prompt": e["caption"], "action_frames": e["action_frames"],
                    "mode": "i2v", "i2v_video": f"{dv_dir}/{d}/gen.mp4", "seed": 1000 + k})

    # ===== video_real → i2v (caption + constant action_label + real first frame) =====
    vr = json.load(open("dataset/videos_real.json"))
    vr_dir = "dataset/video_real"
    vr_map = _id_to_folder(vr_dir)
    vr_ids = [i for i in range(len(vr)) if i in vr_map and vr[i].get("caption")]
    selvr = rnd.sample(vr_ids, min(a.n, len(vr_ids)))
    vrc = []
    for k, i in enumerate(selvr):
        e, d = vr[i], vr_map[i]
        vrc.append({"id": f"vr{i:06d}_{e.get('source_folder', '')}",
                    "prompt": e["caption"], "action_label": int(e["action_label"]),
                    "mode": "i2v", "i2v_video": f"{vr_dir}/{d}/gen.mp4", "seed": 2000 + k})

    os.makedirs(a.out_dir, exist_ok=True)
    for name, data in [("infer_cases.json", t2v), ("infer_cases_i2v.json", i2v),
                       ("infer_cases_video_real.json", vrc)]:
        with open(os.path.join(a.out_dir, name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"  wrote {name}: {len(data)} cases")
    print(f"[build_infer_cases] done (seed={a.seed}, n={a.n})")


if __name__ == "__main__":
    main()
