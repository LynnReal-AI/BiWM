# vendored from minWM HY15 (self-contained: numpy/torch/scipy)
"""
Discrete action utilities for WorldPlay-style action conditioning.

Action representation: action_label = trans_label * 9 + rotate_label (81 classes total)
- Translation (9 classes): no-action(0), forward(1), backward(2), left(3), right(4),
  forward+left(5), forward+right(6), backward+left(7), backward+right(8)
- Rotation (9 classes): same pattern for yaw_right(1), yaw_left(2), pitch_up(3), pitch_down(4), ...
"""

import numpy as np
from scipy.spatial.transform import Rotation


# ─── One-hot to label mapping ──────────────────────────────────────────────────

_TRANS_TABLE = {
    (0, 0, 0, 0): 0,  # No action
    (1, 0, 0, 0): 1,  # Forward only
    (0, 1, 0, 0): 2,  # Backward only
    (0, 0, 1, 0): 3,  # Left only
    (0, 0, 0, 1): 4,  # Right only
    (1, 0, 1, 0): 5,  # Forward + Left
    (1, 0, 0, 1): 6,  # Forward + Right
    (0, 1, 1, 0): 7,  # Backward + Left
    (0, 1, 0, 1): 8,  # Backward + Right
}


def onehot_to_index(one_hot: np.ndarray) -> np.ndarray:
    """Convert (N, 4) one-hot vectors to (N,) integer labels (0-8)."""
    labels = np.zeros(len(one_hot), dtype=np.int64)
    for i, row in enumerate(one_hot):
        key = tuple(int(x) for x in row)
        labels[i] = _TRANS_TABLE.get(key, 0)
    return labels


# ─── Discretize from w2c matrices ──────────────────────────────────────────────

def bin_poses_into_actions(viewmats) -> np.ndarray:
    """
    Derive discrete action labels from w2c view matrices.

    Args:
        viewmats: (T, 4, 4) w2c matrices (already center-normalized to first frame).
                  Accepts numpy array or torch tensor (any dtype).

    Returns:
        actions: (T,) int64 array, action_label = trans_label * 9 + rotate_label
    """
    if not isinstance(viewmats, np.ndarray):
        viewmats = viewmats.float().numpy()
    T = len(viewmats)
    # Convert to c2w
    c2ws = np.linalg.inv(viewmats)

    # Compute relative transforms between consecutive frames
    trans_one_hot = np.zeros((T, 4), dtype=np.int32)  # [forward, backward, left, right]
    rotate_one_hot = np.zeros((T, 4), dtype=np.int32)  # [yaw_right, yaw_left, pitch_up, pitch_down]

    move_norm_threshold = 0.01

    for i in range(1, T):
        # Relative c2w: how frame i moved relative to frame i-1
        rel_c2w = np.linalg.inv(c2ws[i - 1]) @ c2ws[i]
        move_dirs = rel_c2w[:3, 3]
        move_norm = np.linalg.norm(move_dirs)

        # Translation classification
        if move_norm > move_norm_threshold:
            move_norm_dirs = move_dirs / move_norm
            angles_rad = np.arccos(np.clip(move_norm_dirs, -1.0, 1.0))
            angles_deg = np.degrees(angles_rad)

            # Z-axis: forward (< 60°) / backward (> 120°)
            if angles_deg[2] < 60:
                trans_one_hot[i, 0] = 1  # forward
            elif angles_deg[2] > 120:
                trans_one_hot[i, 1] = 1  # backward

            # X-axis: left (> 120°) / right (< 60°)
            if angles_deg[0] < 60:
                trans_one_hot[i, 2] = 1  # left
            elif angles_deg[0] > 120:
                trans_one_hot[i, 3] = 1  # right

        # Rotation classification
        R_rel = rel_c2w[:3, :3]
        r = Rotation.from_matrix(R_rel)
        rot_angles_deg = r.as_euler('xyz', degrees=True)

        rot_threshold = 5e-2  # degrees
        # Yaw (Y-axis rotation)
        if rot_angles_deg[1] > rot_threshold:
            rotate_one_hot[i, 0] = 1  # yaw right
        elif rot_angles_deg[1] < -rot_threshold:
            rotate_one_hot[i, 1] = 1  # yaw left

        # Pitch (X-axis rotation)
        if rot_angles_deg[0] > rot_threshold:
            rotate_one_hot[i, 2] = 1  # pitch up
        elif rot_angles_deg[0] < -rot_threshold:
            rotate_one_hot[i, 3] = 1  # pitch down

    trans_labels = onehot_to_index(trans_one_hot)
    rotate_labels = onehot_to_index(rotate_one_hot)
    actions = trans_labels * 9 + rotate_labels

    return actions
