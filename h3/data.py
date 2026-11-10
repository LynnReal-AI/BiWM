"""A small, explicit preencoded camera dataset; no experiment-runtime imports."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset


def validate_sample(sample):
    required = {"target_latents", "anchor_latents", "prompt_embeds", "text_token_tags",
                "camera_viewmats", "camera_K"}
    if set(sample) != required:
        raise ValueError(f"Expected only {sorted(required)}, got {sorted(sample)}")
    target = sample["target_latents"]
    if target.ndim != 4 or target.shape[0] != 24:
        raise ValueError("target_latents must be [24,T,H,W]")
    _, frames, height, width = target.shape
    if min(height, width) < 2 or height % 2 or width % 2 or frames < 7 or (frames - 2) % 5:
        raise ValueError("Use complete native H3 sequences (T=5n+2), even latent H/W")
    if sample["anchor_latents"].shape != (24, 1, height, width):
        raise ValueError("Encode the RGB first frame independently as anchor_latents")
    prompt = sample["prompt_embeds"]
    tags = sample["text_token_tags"]
    if prompt.ndim != 2 or prompt.shape[1] != 5120 or tags.shape != prompt.shape[:1]:
        raise ValueError("Expected native Qwen embeddings [N,5120] and tags [N]")
    if tags.dtype != torch.int64 or not ((tags == 0) | (tags == 1)).all():
        raise ValueError("Preserve Qwen's image/video=0 and text=1 modality tags")
    if not len(tags):
        raise ValueError("Qwen first-image/caption conditioning is required")
    views, intrinsics = sample["camera_viewmats"], sample["camera_K"]
    if views.shape != (frames, 4, 4) or intrinsics.shape != (frames, 3, 3):
        raise ValueError("Camera tensors must be latent-aligned [T,4,4]/[T,3,3]")
    for name, value in sample.items():
        if name != "text_token_tags" and (not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain finite floating values")
    identity = torch.eye(4, dtype=views.dtype)
    if not torch.allclose(views[0], identity, atol=1e-4, rtol=0):
        raise ValueError("camera_viewmats must be first-frame-relative W2C, before logd4")
    rotation = views[:, :3, :3].float()
    if (not torch.allclose(views[:, 3].float(), identity[3].float().expand(frames, -1), atol=1e-4)
        or not torch.allclose(rotation @ rotation.transpose(-1, -2), torch.eye(3).expand(frames, -1, -1), atol=1e-3)
        or not torch.allclose(torch.linalg.det(rotation), torch.ones(frames), atol=1e-3)):
        raise ValueError("Camera W2C must be rigid SE(3)")
    if not (intrinsics[:, (0, 1), (0, 1)] > 0).all():
        raise ValueError("Camera focal lengths must be positive")
    return sample


class CameraDataset(Dataset):
    """JSONL rows: {\"sample_id\": \"clip\", \"path\": \"relative/sample.safetensors\"}."""
    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError("Empty H3 camera dataset")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sample = load_file(str(self.manifest.parent / row["path"]))
        return validate_sample(sample)
