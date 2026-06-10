"""
Camera PRoPE Dataset for HunyuanVideo AR training.

Loads pre-encoded .pt files (from preencode_camera_video_latents.py),
constructs viewmats (w2c 4x4) and K (3x3 intrinsics) for PRoPE attention.
Poses stored as w2c (OpenCV): [tx, ty, tz, qx, qy, qz, qw].
"""

import json
import os
import sys

sys.path.append(os.path.abspath('.'))
import torch
import numpy as np
import random
import torch.multiprocessing as mp

from torch.utils.data import Dataset, Sampler, DataLoader
from scipy.spatial.transform import Rotation

# vendored from minWM HY15 ar_camera_plucker_dataset; self-contained within pipelines
# (torch.distributed + local action_utils, no trainer.* dependency).
import logging
import torch.distributed as dist

from pipelines.dataset.hy15_action_utils import bin_poses_into_actions

log_handle = logging.getLogger("hy15_camera_dataset")


def _distributed_ready():
    return dist.is_available() and dist.is_initialized()


def fetch_world_size():
    return dist.get_world_size() if _distributed_ready() else 1


def fetch_world_rank():
    return dist.get_rank() if _distributed_ready() else 0


def fetch_sp_world_size():
    # sequence-parallel group size: default 1 (no SP); if SP is enabled it is specified by the env var HY15_SP_SIZE.
    return int(os.environ.get("HY15_SP_SIZE", "1"))


def fetch_local_torch_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    return torch.device("cpu")


# ─── Pose interpolation & camera matrix construction ─────────────────────────


def make_viewmats_and_intrinsics(intrinsics: np.ndarray, poses: np.ndarray):
    """
    Build 4x4 w2c view matrices and 3x3 intrinsics matrices from poses.
    Applies camera center normalization (align to first frame).

    Args:
        intrinsics: (4,) [fx, fy, cx, cy] normalized
        poses:      (T, 7) [tx, ty, tz, qx, qy, qz, qw] w2c (OpenCV)

    Returns:
        viewmats: (T, 4, 4) w2c SE3 matrices (normalized to first frame)
        Ks:       (T, 3, 3) intrinsics matrices (normalized)
    """
    T = len(poses)
    fx, fy, cx, cy = intrinsics

    viewmats = np.zeros((T, 4, 4), dtype=np.float32)
    for i in range(T):
        tx, ty, tz, qx, qy, qz, qw = poses[i]
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # R_w2c
        viewmats[i, :3, :3] = R
        viewmats[i, :3, 3] = [tx, ty, tz]
        viewmats[i, 3, 3] = 1.0

    # Camera center normalization: align all poses to first frame
    c2w = np.linalg.inv(viewmats)
    C0_inv = np.linalg.inv(c2w[0])  # = viewmats[0]
    c2w_aligned = np.array([C0_inv @ C for C in c2w])
    viewmats = np.linalg.inv(c2w_aligned).astype(np.float32)

    # K matrix (same for all frames) - normalized focal lengths
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    Ks = np.tile(K, (T, 1, 1))  # (T, 3, 3)

    return viewmats, Ks


# ─── Translation normalization ──────────────────────────────────────────────

def rescale_translations(viewmats: np.ndarray):
    """Normalize translation part of viewmats. Rotations untouched."""
    return viewmats, False


# ─── Sampler (reuse from existing) ───────────────────────────────────────────

class DataSeqParallelBatchPicker(Sampler[list[int]]):
    def __init__(self, batch_size, dataset_size, num_sp_groups, sp_world_size,
                 global_rank, drop_last=True, drop_first_row=False, seed=0):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        rng = torch.Generator().manual_seed(seed)
        global_indices = torch.randperm(self.dataset_size, generator=rng)

        if drop_first_row:
            global_indices = global_indices[global_indices != 0]
            self.dataset_size -= 1

        if drop_last:
            num_batches = self.dataset_size // self.batch_size
            num_global_batches = num_batches // num_sp_groups
            global_indices = global_indices[:num_global_batches * num_sp_groups * self.batch_size]
        else:
            remainder = self.dataset_size % (num_sp_groups * self.batch_size)
            if remainder:
                padding = num_sp_groups * self.batch_size - remainder
                global_indices = torch.cat([global_indices, global_indices[:padding]])

        ith_sp_group = global_rank // sp_world_size
        self.sp_group_local_indices = global_indices[ith_sp_group::num_sp_groups]
        log_handle.info("Dataset size for each sp group: %d", len(self.sp_group_local_indices))

    def __iter__(self):
        for i in range(0, len(self.sp_group_local_indices), self.batch_size):
            yield self.sp_group_local_indices[i:i + self.batch_size].tolist()

    def __len__(self):
        return len(self.sp_group_local_indices) // self.batch_size


# ─── Dataset ─────────────────────────────────────────────────────────────────

class CameraPluckerData(Dataset):
    """
    Loads pre-encoded .pt with camera data, constructs viewmats/Ks for PRoPE.

    Batch keys:
        latent, prompt_embed, prompt_mask, byt5_text_states, byt5_text_mask,
        image_cond, vision_states, i2v_mask, viewmats, Ks,
        video_path, select_window_out_flag
    """

    def __init__(self, json_path, causal, window_frames, batch_size, cfg_rate,
                 i2v_rate, task_type, drop_last, drop_first_row, seed, device,
                 shared_state, latent_spatial_scale=8):
        self.json_data = json.load(open(json_path, 'r'))
        self.all_length = len(self.json_data)
        self.causal = causal
        self.window_frames = window_frames
        self.cfg_rate = cfg_rate
        self.rng = random.Random(seed)
        self.i2v_rate = i2v_rate
        self.task_type = task_type
        self.device = device
        self.shared_state = shared_state
        self.latent_spatial_scale = latent_spatial_scale

        self.sampler = DataSeqParallelBatchPicker(
            batch_size=batch_size,
            dataset_size=self.all_length,
            num_sp_groups=fetch_world_size() // fetch_sp_world_size(),
            sp_world_size=fetch_sp_world_size(),
            global_rank=fetch_world_rank(),
            drop_last=drop_last,
            drop_first_row=drop_first_row,
            seed=seed,
        )

        neg_prompt_path = os.environ.get(
            "NEG_PROMPT_PT", os.path.join(os.path.dirname(json_path), "hunyuan_neg_prompt.pt"))
        neg_byt5_path = os.environ.get(
            "NEG_BYT5_PT", os.path.join(os.path.dirname(json_path), "hunyuan_neg_byt5_prompt.pt"))

        self.neg_prompt_pt = torch.load(neg_prompt_path, map_location="cpu", weights_only=True)
        self.neg_byt5_pt = torch.load(neg_byt5_path, map_location="cpu", weights_only=True)

    def __len__(self):
        return self.all_length
    
    def update_max_frames(self, new_max_frames):
        pass

    def __getitem__(self, idx):
        while True:
            try:
                json_data = self.json_data[idx]
                latent_pt_path = json_data['latent_path']

                latent_pt = torch.load(latent_pt_path, map_location="cpu", weights_only=True)
                latent = latent_pt['latent'][0]  # (C, T_lat, H_lat, W_lat)
                latent_length = latent.shape[1]

                if latent_length < self.window_frames:
                    idx = self.rng.randint(0, self.all_length - 1)
                    continue

                max_frames = int(self.shared_state["max_frames"]) // 4 * 4
                max_length = min(max_frames, latent_length // 4 * 4)
                max_length = min(max_length, self.window_frames)
                latent = latent[:, :max_length, ...]

                T_lat = latent.shape[1]
                H_lat = latent.shape[2]
                W_lat = latent.shape[3]

                # ── Camera matrices for PRoPE ──
                intrinsics = latent_pt['intrinsics'].numpy()        # (4,)
                if intrinsics[0] <= 0 or intrinsics[1] <= 0:
                    idx = self.rng.randint(0, self.all_length - 1)
                    continue
                poses_raw = latent_pt['poses'].numpy()              # (N_cam, 7)

                # poses_raw has one pose per latent frame; no interpolation needed
                poses_lat = poses_raw[:T_lat]
                viewmats, Ks = make_viewmats_and_intrinsics(intrinsics, poses_lat)

                # ── Translation normalization ──
                viewmats, discard = rescale_translations(viewmats)
                if discard:
                    idx = self.rng.randint(0, self.all_length - 1)
                    continue

                # ── Discrete action labels ──
                action = torch.from_numpy(
                    bin_poses_into_actions(viewmats))  # (T_lat,)

                viewmats = torch.from_numpy(viewmats)  # (T_lat, 4, 4)
                Ks = torch.from_numpy(Ks)              # (T_lat, 3, 3)

                # ── Text embeddings ──
                prompt_embed = latent_pt['prompt_embeds'][0]
                prompt_mask = latent_pt['prompt_mask'][0]
                byt5_text_states = latent_pt['byt5_text_states'][0]
                byt5_text_mask = latent_pt['byt5_text_mask'][0]

                if self.rng.random() < self.cfg_rate:
                    prompt_embed = self.neg_prompt_pt['negative_prompt_embeds'][0]
                    prompt_mask = self.neg_prompt_pt['negative_prompt_mask'][0]
                    byt5_text_states = self.neg_byt5_pt['byt5_text_states'][0]
                    byt5_text_mask = self.neg_byt5_pt['byt5_text_mask'][0]

                # ── Image conditioning ──
                if self.task_type == "i2v":
                    image_cond = latent_pt['image_cond'][0]
                    vision_states = latent_pt['vision_states'][0]
                else:
                    image_cond = torch.zeros(32, 1, H_lat, W_lat)
                    vision_states = torch.zeros(1, 4096, 3584)

                i2v_mask = torch.ones_like(latent)

                batch = {
                    "latent": latent,
                    "viewmats": viewmats,
                    "Ks": Ks,
                    "action": action,
                    "prompt_embed": prompt_embed,
                    "prompt_mask": prompt_mask,
                    "byt5_text_states": byt5_text_states,
                    "byt5_text_mask": byt5_text_mask,
                    "image_cond": image_cond,
                    "vision_states": vision_states,
                    "i2v_mask": i2v_mask,
                    "video_path": latent_pt_path,
                    "select_window_out_flag": 0,
                }
                break
            except Exception as e:
                print('error:', e, latent_pt_path, flush=True)
                idx = self.rng.randint(0, self.all_length - 1)
        return batch


# ─── Collate & builder ───────────────────────────────────────────────────────

def plucker_batch_merge(batch):
    result = {
        "latent":             torch.stack([b["latent"] for b in batch]),
        "viewmats":           torch.stack([b["viewmats"] for b in batch]),
        "Ks":                 torch.stack([b["Ks"] for b in batch]),
        "action":             torch.stack([b["action"] for b in batch]),
        "prompt_embed":       torch.stack([b["prompt_embed"] for b in batch]),
        "prompt_mask":        torch.stack([b["prompt_mask"] for b in batch]),
        "byt5_text_states":   torch.stack([b["byt5_text_states"] for b in batch]),
        "byt5_text_mask":     torch.stack([b["byt5_text_mask"] for b in batch]),
        "image_cond":         torch.stack([b["image_cond"] for b in batch]),
        "vision_states":      torch.stack([b["vision_states"] for b in batch]),
        "i2v_mask":           torch.stack([b["i2v_mask"] for b in batch]),
        "video_path":         [b["video_path"] for b in batch],
        "select_window_out_flag": [b["select_window_out_flag"] for b in batch],
    }
    return result


def make_camera_plucker_loader(
        json_path, causal, window_frames, batch_size, num_data_workers,
        drop_last, drop_first_row, seed, cfg_rate, i2v_rate, task_type,
) -> tuple:
    manager = mp.Manager()
    shared_state = manager.dict()
    shared_state["max_frames"] = window_frames

    dataset = CameraPluckerData(
        json_path, causal, window_frames, batch_size, cfg_rate, i2v_rate,
        task_type, drop_last=drop_last, drop_first_row=drop_first_row,
        seed=seed, device=fetch_local_torch_device(), shared_state=shared_state,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=plucker_batch_merge,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader
