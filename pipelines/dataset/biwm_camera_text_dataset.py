# BiWM discrete camera-text dataset (dataset/videos)
# Directory structure:  dataset/videos/{6-digit id}_{camera token string}/gen.mp4
# caption + pose live in dataset/videos_syn.json (list, index == 6-digit id):
#   { "caption": "...", "action_frames": "up-4, right-8, s-7" }
# Each "token-N" segment lasts N latent frames; action_labels[0]=0 (init static).
import json
import os
import random
import re
import signal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from decord import VideoReader, cpu
except Exception:  # defer the error when decord is missing
    VideoReader, cpu = None, None


# token → 81-class discrete action_label (consistent with ACTION_TEXT_TABLE in wan/modules/model.py)
TOKEN_LABEL_LOOKUP = {
    # translation (rot=0):  trans*9 + 0
    "w": 9, "s": 18, "d": 27, "a": 36,
    # rotation (trans=0):  0*9 + rot
    "up": 1, "down": 2, "right": 3, "left": 4,
}

_SEGMENT_PATTERN = re.compile(r"([a-zA-Z]+|\d+)\s*-?\s*(\d+)")


def pose_text_to_motion_labels(pose_str: str, num_latent_frames: int) -> torch.Tensor:
    """Parse "up-4, right-8, s-7" into a [num_latent_frames] int discrete label tensor (0~80).

    The 0th latent frame is the init frame, label fixed to 0 (static). The rest are expanded segment by segment.
    If the expanded length does not match num_latent_frames, pad with the last label / truncate at the tail.
    """
    labels = [0]  # init frame: static
    for m in _SEGMENT_PATTERN.finditer(pose_str):
        tok = m.group(1).lower()
        n = int(m.group(2))
        # token: named single-axis (videos_syn) or numeric 81-class label (videos_real combined action)
        if tok.isdigit():
            lab = int(tok)
            if not (0 <= lab <= 80):
                raise ValueError(f"action_label {lab} out of range [0,80] in action_frames='{pose_str}'")
        elif tok in TOKEN_LABEL_LOOKUP:
            lab = TOKEN_LABEL_LOOKUP[tok]
        else:
            raise ValueError(f"unknown camera token '{tok}' in action_frames='{pose_str}'")
        labels.extend([lab] * n)

    if len(labels) < num_latent_frames:
        pad_val = labels[-1] if labels else 0
        labels.extend([pad_val] * (num_latent_frames - len(labels)))
    labels = labels[:num_latent_frames]
    return torch.tensor(labels, dtype=torch.long)


def biwm_collate(batch):
    """Unified collate for batch_size=1: return the single sample dict (no batch dim)."""
    return batch[0]


class BiwmCamCaptionData(Dataset):
    """dataset/videos discrete camera-text dataset.

    __getitem__ returns a dict (consumed by both wan and hy15 via keys; use collate_fn=lambda b: b[0],
    batch_size=1, so each sample has no batch dim):
        pixel_values[T,C,H,W], ref_img[C,H,W], caption, video_id, has_camera, source,
        caption_type, frame_start, frame_end, action_labels[t_lat].
    """

    VIDEO_READ_TIMEOUT = 120

    def __init__(
        self,
        video_dir: str,
        caption_json: str,
        width: int,
        height: int,
        num_frames: int = 77,
        vae_temporal_factor: int = 4,
        video_filename: str = "gen.mp4",
        max_samples: int = None,
    ):
        super().__init__()
        if VideoReader is None:
            raise ImportError("decord not installed, cannot read videos")
        self.video_dir = video_dir
        self.width = int(width)
        self.height = int(height)
        self.num_frames = int(num_frames)
        self.vae_temporal_factor = int(vae_temporal_factor)
        self.video_filename = video_filename
        self.num_latent_frames = (self.num_frames - 1) // self.vae_temporal_factor + 1
        self._to_tensor = transforms.ToTensor()

        # read the caption + pose table (index == id)
        with open(caption_json, "r", encoding="utf-8") as f:
            self._meta = json.load(f)

        # scan subdirectories that contain gen.mp4, aligned with meta
        self.samples = []  # (mp4_path, caption, pose_str, video_id)
        missing_meta = 0
        for name in sorted(os.listdir(video_dir)):
            sub = os.path.join(video_dir, name)
            if not os.path.isdir(sub):
                continue
            mp4 = os.path.join(sub, self.video_filename)
            if not os.path.exists(mp4):
                continue
            try:
                vid_id = int(name.split("_")[0])
            except ValueError:
                continue
            if vid_id < 0 or vid_id >= len(self._meta):
                missing_meta += 1
                continue
            entry = self._meta[vid_id]
            caption = entry.get("caption", "")
            # action_frames: unified camera field "<token>-<n>" (token = named single-axis for videos_syn,
            #   or numeric 81-class label for videos_real combined actions); each segment lasts n latent frames.
            #   (falls back to the legacy "pose_str" key for backward compatibility.)
            pose_str = entry.get("action_frames", entry.get("pose_str", ""))
            action_label = entry.get("action_label", None)
            if not pose_str and action_label is None:
                missing_meta += 1
                continue
            self.samples.append((mp4, caption, pose_str, name, action_label))

        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        print(
            f"[BiwmCamCaptionData] {len(self.samples)} samples from {video_dir} "
            f"(missing_meta={missing_meta}), num_frames={self.num_frames}, "
            f"latent_frames={self.num_latent_frames}, size={self.height}x{self.width}",
            flush=True,
        )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _alarm_handler(signum, frame):
        raise TimeoutError("video read timeout")

    def _process_frame(self, img: Image.Image) -> torch.Tensor:
        tensor = self._to_tensor(img)  # [C,H,W] in [0,1]
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=(self.height, self.width),
            mode="bicubic", align_corners=False, antialias=True,
        )
        return tensor.squeeze(0)

    def _sample_frame_ids(self, video_length: int):
        """Sample num_frames frame ids, satisfying the VAE temporal constraint (F-1)%vtf==0."""
        n = self.num_frames
        if video_length >= n:
            # over-long clips: take the first n consecutive frames (native frame rate)
            ids = list(range(n))
        else:
            # if insufficient, loop to fill up
            ids = list(range(video_length))
            while len(ids) < n:
                ids.append(ids[-1])
            ids = ids[:n]
        # align to (F-1) % vtf == 0
        vtf = self.vae_temporal_factor
        if (len(ids) - 1) % vtf != 0:
            valid = ((len(ids) - 1) // vtf) * vtf + 1
            ids = ids[:valid]
        return ids

    def _read_video(self, mp4_path: str):
        vr = VideoReader(mp4_path, ctx=cpu(0))
        video_length = len(vr)
        frame_ids = self._sample_frame_ids(video_length)
        batch = vr.get_batch(frame_ids).asnumpy()  # [T,H,W,C] uint8
        frames = [Image.fromarray(batch[i]) for i in range(batch.shape[0])]
        return frames

    def __getitem__(self, idx):
        max_retries = 10
        for retry in range(max_retries):
            mp4_path, caption, pose_str, vid_id, action_label = self.samples[idx]
            old_handler = signal.signal(signal.SIGALRM, self._alarm_handler)
            signal.alarm(self.VIDEO_READ_TIMEOUT)
            try:
                frames = self._read_video(mp4_path)
                if len(frames) == 0:
                    raise RuntimeError(f"no frames read from {mp4_path}")

                pixel_values = torch.stack([self._process_frame(f) for f in frames], dim=0)  # [T,C,H,W] [0,1]
                pixel_values = pixel_values.sub_(0.5).div_(0.5)  # -> [-1,1]
                ref_img = pixel_values[0]

                # latent frame count varies with the actual frame count (after alignment)
                t_lat = (pixel_values.shape[0] - 1) // self.vae_temporal_factor + 1
                if action_label is not None:
                    # ★ video_real: whole clip constant single action. init frame static(0), the rest of the latent frames = combined label.
                    _lab = [0] + [int(action_label)] * max(0, t_lat - 1)
                    action_labels = torch.tensor(_lab[:t_lat], dtype=torch.long)
                else:
                    action_labels = pose_text_to_motion_labels(pose_str, t_lat)

                # Unified named sample (discrete camera only; consumed by both wan and hy15 via keys)
                return {
                    "pixel_values": pixel_values,        # [T,C,H,W] in [-1,1]
                    "ref_img": ref_img,                  # [C,H,W]
                    "caption": caption,                  # str
                    "video_id": vid_id,                  # str
                    "has_camera": True,
                    "source": "biwm",
                    "caption_type": "overall",
                    "frame_start": 0,
                    "frame_end": int(pixel_values.shape[0]),
                    "action_labels": action_labels,      # [t_lat] discrete 81-class labels
                }
            except Exception as e:
                new_idx = random.randint(0, len(self.samples) - 1)
                print(
                    f"[BiwmCamCaptionData] __getitem__ retry {retry}/{max_retries}: "
                    f"{type(e).__name__}: {e} | video={mp4_path} | reroll to {new_idx}",
                    flush=True,
                )
                idx = new_idx
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        # all failed → skip placeholder (compatible with the main loop's skip detection)
        print(f"[BiwmCamCaptionData] all {max_retries} retries failed, returning skip", flush=True)
        return {"skip": True}


# Pre-encoded dataset: read the .pt produced by preencode_hy15.py → training dict
# (used by --data_mode preenc, skips per-step online VAE+MLLM encoding).
import glob as _glob

_BYT5_MAX_TOKENS = 256
_BYT5_WIDTH = 1472


class BiwmCachedFeatureData(torch.utils.data.Dataset):
    """Read the per-clip .pt output by preencode_hy15.py, return a dict consistent with run_one_step.
    caption_only=True (stage2 DMD training): use prompt_embed_capt (pure caption); otherwise use prompt_embed (<camera> text).
    byt5/vision/image_cond are filled with zeros (t2v); viewmats/Ks/action not provided (camtext). collate uses _live_batch_merge (returns b[0])."""

    def __init__(self, preenc_dir, caption_only=False):
        self.files = sorted(_glob.glob(os.path.join(preenc_dir, "*.pt")))
        if not self.files:
            raise FileNotFoundError(f"BiwmCachedFeatureData: no .pt under {preenc_dir}")
        self.caption_only = caption_only
        print(f"[BiwmCachedFeatureData] {preenc_dir}: {len(self.files)} pre-encoded clips "
              f"(caption_only={caption_only})", flush=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        for _ in range(8):
            try:
                rec = torch.load(self.files[idx], map_location="cpu", weights_only=False)
                latent = rec["latent"]                         # [1,C,T,h,w] bf16
                if self.caption_only and "prompt_embed_capt" in rec:
                    pe, pm = rec["prompt_embed_capt"], rec["prompt_mask_capt"]
                else:
                    pe, pm = rec["prompt_embed"], rec["prompt_mask"]
                _, C, T, h, w = latent.shape
                bf16 = torch.bfloat16
                return {
                    "latent": latent.to(bf16),
                    "prompt_embed": pe.to(bf16),
                    "prompt_mask": pm.to(bf16),
                    "byt5_text_states": torch.zeros(1, _BYT5_MAX_TOKENS, _BYT5_WIDTH, dtype=bf16),
                    "byt5_text_mask": torch.zeros(1, _BYT5_MAX_TOKENS, dtype=bf16),
                    "vision_states": torch.zeros(1, 1, 1, dtype=bf16),
                    "image_cond": torch.zeros(1, C, 1, h, w, dtype=bf16),
                    "i2v_mask": torch.ones_like(latent, dtype=bf16),
                    "caption": rec.get("caption", ""),
                    "clip_action": int(rec.get("action_label", 0)),
                    # camtext: do not provide viewmats/Ks/action (run_one_step uses batch.get → None)
                }
            except Exception as e:
                import random as _r
                print(f"[BiwmCachedFeatureData] load {self.files[idx]} failed: {type(e).__name__}: {e}, reroll",
                      flush=True)
                idx = _r.randint(0, len(self.files) - 1)
        raise RuntimeError("BiwmCachedFeatureData: consecutive load failures")
