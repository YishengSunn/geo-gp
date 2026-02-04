import csv
import numpy as np

from geometry.so3 import so3_log, so3_exp


# ============================================================
# Standardization helpers
# ============================================================

class Standardizer:
    def fit(self, X, Y):
        """
        Fit standardizer to data

        Args:
            X: input data, torch tensor of shape (N, D_in)
            Y: output data, torch tensor of shape (N, D_out)
            
        Returns:
            self
        """
        self.X_mean = X.mean(0)  # Shape: (D_in,)
        self.X_std = X.std(0).clamp_min(1e-8)  # Shape: (D_in,)
        self.Y_mean = Y.mean(0)  # Shape: (D_out,)
        self.Y_std = Y.std(0).clamp_min(1e-8)  # Shape: (D_out,)

        return self

    def x_transform(self, X): return (X - self.X_mean) / self.X_std

    def y_transform(self, Y): return (Y - self.Y_mean) / self.Y_std

    def y_inverse_transform(self, Yn):
        """
        Inverse transform standardized output
        Args:
            Yn: standardized output, torch tensor of shape (..., D_out)
        Returns:
            Y: original output, torch tensor of shape (..., D_out)
        """
        assert Yn.shape[-1] == self.Y_std.shape[0], f"维度不匹配: Yn.shape={Yn.shape}, std={self.Y_std.shape}"

        return Yn * self.Y_std + self.Y_mean

    # Compatible with old interface
    def y_inverse(self, Yn): return self.y_inverse_transform(Yn)

# ============================================================
# Tensor/Array conversion helpers
# ============================================================

def torch_to_np(x):
    """
    Convert torch tensor to numpy array
    
    Args:
        x: torch tensor
    Returns:
        numpy array
    """
    return x.detach().cpu().numpy()

# ============================================================
# Geometric helpers
# ============================================================

def closest_index(pt, arr):
    """
    Find the index of the point in arr that is closest to pt.
    
    Args:
        pt: (2,) array-like, target point
        arr: (N,2) array-like, array of points to search

    Returns:
        idx: int, index of the closest point in arr
    """
    arr = np.asarray(arr, dtype=np.float64)
    d = np.linalg.norm(arr - pt[None, :], axis=1)

    return int(np.argmin(d))

# ============================================================
# Smoothing helpers (post-processing)
# ============================================================

def moving_average_centered(arr, win):
    """
    Centered moving average smoothing along time axis (axis=0).

    Args:
        arr: (N, d) array-like
        win: int, window size (odd preferred)

    Returns:
        out: (N, d) smoothed array
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]

    win = int(win)
    if win < 3:
        return arr
    if win % 2 == 0:
        win += 1

    pad = win // 2
    if arr.shape[0] <= 1:
        return arr

    # Reflect padding along time axis
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="reflect")
    kernel = np.ones(win, dtype=np.float64) / float(win)

    out = np.empty_like(arr, dtype=np.float64)
    for d in range(arr.shape[1]):
        out[:, d] = np.convolve(padded[:, d], kernel, mode="valid")
        
    return out

def moving_average_centered_6d(arr, win):
    """
    Centered moving average smoothing along time axis (axis=0).

    Supports:
        - (N, d) Euclidean features (position, spherical, etc.)
        - (N, 3, 3) rotation matrices (SO(3)) via log-exp smoothing

    Args:
        arr: array-like
        win: int, window size (odd preferred)

    Returns:
        out: smoothed array, same shape as input
    """
    arr = np.asarray(arr, dtype=np.float64)

    # Rotation
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        # Convert rotations to so(3) vectors
        omegas = np.array([so3_log(R) for R in arr])  # (N,3)

        # Smooth in R^3
        omegas_s = moving_average_centered(omegas, win)

        # Map back to SO(3)
        R_s = np.array([so3_exp(w) for w in omegas_s])
        return R_s

    # Position
    if arr.ndim == 1:
        arr = arr[:, None]

    win = int(win)
    if win < 3:
        return arr
    if win % 2 == 0:
        win += 1

    pad = win // 2
    if arr.shape[0] <= 1:
        return arr

    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="reflect")
    kernel = np.ones(win, dtype=np.float64) / float(win)

    out = np.empty_like(arr, dtype=np.float64)
    for d in range(arr.shape[1]):
        out[:, d] = np.convolve(padded[:, d], kernel, mode="valid")

    return out

def smooth_prediction_by_velocity(
    probe: np.ndarray,
    pred: np.ndarray,
    *,
    win: int = 9,
    blend_first_step: float = 0.8,
) -> np.ndarray:
    """
    Smooth predicted trajectory by smoothing per-step velocity (delta) in world/probe frame.

    This keeps the first predicted point continuous w.r.t. the last probe point, and blends the
    first velocity to preserve the probe's exiting direction.

    Args:
        probe_xy: (Np, D) probe points in world/probe frame
        pred_xy: (Hp, D) predicted points in world/probe frame (future, starting AFTER probe end)
        win: centered moving-average window (odd preferred)
        blend_first_step: blend factor for the first predicted delta:
            v0_smooth = blend_first_step * v0_probe + (1-blend_first_step) * v0_pred

    Returns:
        pred_xy_smooth: (Hp, D) smoothed predicted points in world/probe frame
    """
    probe = np.asarray(probe, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    if pred.size == 0:
        return pred
    if probe.shape[0] < 2:
        return pred

    # Build a full polyline that includes the last probe point as the boundary
    p_last = probe[-1]
    full = np.vstack([p_last[None, :], pred])  # (Hp+1, D)

    # Velocity sequence for the predicted part
    v = np.diff(full, axis=0)  # (Hp, D)

    # Smooth velocities using a centered moving average
    v_s = moving_average_centered(v, int(win))

    # Enforce a smooth connection direction using the probe's last delta
    v0_probe = probe[-1] - probe[-2]
    if np.linalg.norm(v0_probe) > 1e-12:
        v_s[0] = float(blend_first_step) * v0_probe + (1.0 - float(blend_first_step)) * v_s[0]

    # Reconstruct smoothed prediction points
    out = np.empty_like(pred, dtype=np.float64)
    cur = p_last.copy()
    for i in range(pred.shape[0]):
        cur = cur + v_s[i]
        out[i] = cur

    return out

def smooth_prediction_by_twist_6d(
    probe_pos: np.ndarray,
    probe_rot: np.ndarray,
    pred_pos: np.ndarray,
    pred_rot: np.ndarray,
    *,
    win: int = 9,
    blend_first_step_pos: float = 0.8,
    blend_first_step_rot: float = 0.8,
    eps: float = 1e-12,
):
    """
    Smooth 6D predicted trajectory by smoothing per-step twist in WORLD/PROBE frame:
      - translation part: smooth Δp (world-frame)
      - rotation part: smooth ω where Exp(ω) = R_{t}^T R_{t+1}  (body-frame incremental rotation)

    This keeps continuity w.r.t. the last probe pose and blends the first step to preserve
    the exiting direction (both translation and rotation).

    Args:
        probe_pos: (Np, 3) probe positions up to current time
        probe_rot: (Np, 3, 3) probe rotations up to current time
        pred_pos:  (Hp, 3) predicted future positions (starting AFTER probe end)
        pred_rot:  (Hp, 3, 3) predicted future rotations (starting AFTER probe end)
        win: centered moving-average window (odd preferred)
        blend_first_step_pos: blend factor for first Δp
        blend_first_step_rot: blend factor for first ω
        eps: numerical threshold

    Returns:
        pred_pos_s: (Hp, 3) smoothed predicted positions
        pred_rot_s: (Hp, 3, 3) smoothed predicted rotations (still in SO(3))
    """
    probe_pos = np.asarray(probe_pos, dtype=np.float64)
    pred_pos  = np.asarray(pred_pos,  dtype=np.float64)
    probe_rot = np.asarray(probe_rot, dtype=np.float64)
    pred_rot  = np.asarray(pred_rot,  dtype=np.float64)

    Hp = pred_pos.shape[0]
    if Hp == 0:
        return pred_pos, pred_rot
    if probe_pos.shape[0] < 2 or probe_rot.shape[0] < 2:
        return pred_pos, pred_rot

    # Translation: smooth Δp
    p_last = probe_pos[-1]
    full_p = np.vstack([p_last[None, :], pred_pos])  # (Hp+1, 3)
    v = np.diff(full_p, axis=0)                      # (Hp, 3)

    v_s = moving_average_centered(v, int(win))

    # Blend first step with probe's last delta
    v0_probe = probe_pos[-1] - probe_pos[-2]
    if np.linalg.norm(v0_probe) > eps:
        v_s[0] = float(blend_first_step_pos) * v0_probe + (1.0 - float(blend_first_step_pos)) * v_s[0]

    pred_pos_s = np.empty_like(pred_pos)
    cur_p = p_last.copy()
    for i in range(Hp):
        cur_p = cur_p + v_s[i]
        pred_pos_s[i] = cur_p

    # Rotation: smooth ω from relative rotations
    # ω_i = Log( R_{i}^T R_{i+1} )
    R_last = probe_rot[-1]
    full_R = np.concatenate([R_last[None, :, :], pred_rot], axis=0)  # (Hp+1, 3, 3)

    omegas = np.empty((Hp, 3), dtype=np.float64)
    for i in range(Hp):
        dR = full_R[i].T @ full_R[i+1]
        omegas[i] = so3_log(dR)

    omegas_s = moving_average_centered(omegas, int(win))

    # Blend first step with probe's last angular increment
    dR_probe = probe_rot[-2].T @ probe_rot[-1]
    omega0_probe = so3_log(dR_probe)
    if np.linalg.norm(omega0_probe) > eps:
        omegas_s[0] = float(blend_first_step_rot) * omega0_probe + (1.0 - float(blend_first_step_rot)) * omegas_s[0]

    pred_rot_s = np.empty_like(pred_rot)
    cur_R = R_last.copy()
    for i in range(Hp):
        cur_R = cur_R @ so3_exp(omegas_s[i])
        pred_rot_s[i] = cur_R

    return pred_pos_s, pred_rot_s

def load_traj_all_csv(path):
    """
    Load trajectories from traj_all_*.csv exported by save_csv.

    Returns:
        ref_pts:   (N,2) np.ndarray
        sampled:   (M,2) np.ndarray
        probe_pts: (K,2) np.ndarray
    """
    ref_pts = []
    sampled = []
    probe_pts = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Reference trajectory
            if row["ref_x"] != "" and row["ref_y"] != "":
                ref_pts.append([float(row["ref_x"]), float(row["ref_y"])])

            # Sampled points
            if row["sampled_x"] != "" and row["sampled_y"] != "":
                sampled.append([float(row["sampled_x"]), float(row["sampled_y"])])

            # Probe points
            if row["probe_x"] != "" and row["probe_y"] != "":
                probe_pts.append([float(row["probe_x"]), float(row["probe_y"])])

    return (
        np.asarray(ref_pts, dtype=np.float64),
        np.asarray(sampled, dtype=np.float64),
        np.asarray(probe_pts, dtype=np.float64),
    )
