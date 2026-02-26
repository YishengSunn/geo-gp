import csv
import pandas as pd
import numpy as np

from utils.quaternion import quat_mul, quat_inv, quat_normalize, quat_log, quat_exp
from utils.so3 import so3_log, so3_exp


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
# Smoothing helpers (post-processing)
# ============================================================

def moving_average_centered_pos(arr, win):
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

def moving_average_centered_orient(arr, win):
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
    N, D = arr.shape

    if N <= win:
        return arr.copy()

    # Reflect padding along time axis
    kernel = np.ones(win, dtype=np.float64) / float(win)
    out = arr.copy()

    for d in range(D):
        smoothed = np.convolve(arr[:, d], kernel, mode='valid')
        out[pad:N-pad, d] = smoothed
        
    return out

def moving_average_centered_6d(arr, win):
    """
    Centered moving average smoothing along time axis (axis=0).

    Supports:
    - (N, d) Euclidean features (position, spherical, etc.)
    - (N, 4) quaternions via log-exp smoothing

    Args:
        arr: array-like
        win: int, window size (odd preferred)

    Returns:
        out: smoothed array, same shape as input
    """
    arr = np.asarray(arr, dtype=np.float64)

    # Quaternion
    if arr.ndim == 2 and arr.shape[1] == 4:
        # Ensure sign continuity
        qs = arr.copy()
        for i in range(1, len(qs)):
            if np.dot(qs[i-1], qs[i]) < 0:
                qs[i] = -qs[i]

        # Log map
        omegas = np.array([quat_log(q) for q in qs])

        # Smooth in R^3
        omegas_s = moving_average_centered_orient(omegas, win)

        # Exp map back
        qs_s = np.array([quat_exp(w) for w in omegas_s])

        # Normalize
        qs_s = np.array([quat_normalize(q) for q in qs_s])

        return qs_s

    # Euclidean
    else:
        return moving_average_centered_pos(arr, win)

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
    v_s = moving_average_centered_pos(v, int(win))

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
    probe_quat: np.ndarray,
    pred_pos: np.ndarray,
    pred_quat: np.ndarray,
    *,
    win: int = 9,
    blend_first_step_pos: float = 0.8,
    blend_first_step_rot: float = 0.8,
    eps: float = 1e-12,
):
    """
    Smooth 6D predicted trajectory by smoothing per-step twist in WORLD/PROBE frame, using quaternions for rotation:
        - translation part: smooth Δp (world-frame)
        - rotation part: smooth ω where Exp(ω) = quat_inv(q_{t}) * q_{t+1}  (body-frame incremental rotation)
    
    This keeps continuity w.r.t. the last probe pose and blends the first step to preserve the exiting direction (both translation and rotation).

    Args:
        probe_pos: (Np, 3) probe positions up to current time
        probe_quat: (Np, 4) probe rotations as quaternions up to current time
        pred_pos: (Hp, 3) predicted future positions (starting after probe end)
        pred_quat: (Hp, 4) predicted future rotations as quaternions (starting after probe end)
        win: centered moving-average window (odd preferred)
        blend_first_step_pos: blend factor for first Δp
        blend_first_step_rot: blend factor for first ω
        eps: numerical threshold

    Returns:
        pred_pos_s: (Hp, 3) smoothed predicted positions
        pred_quat_s: (Hp, 4) smoothed predicted rotations as quaternions (still normalized)
    """
    probe_pos = np.asarray(probe_pos, dtype=np.float64)
    pred_pos  = np.asarray(pred_pos,  dtype=np.float64)
    probe_quat = np.asarray(probe_quat, dtype=np.float64)
    pred_quat  = np.asarray(pred_quat,  dtype=np.float64)

    Hp = pred_pos.shape[0]

    if Hp == 0:
        return pred_pos, pred_quat

    if probe_pos.shape[0] < 2 or probe_quat.shape[0] < 2:
        return pred_pos, pred_quat

    # Translation smoothing
    p_last = probe_pos[-1]
    full_p = np.vstack([p_last[None, :], pred_pos])
    v = np.diff(full_p, axis=0)

    v_s = moving_average_centered_pos(v, int(win))

    v0_probe = probe_pos[-1] - probe_pos[-2]
    if np.linalg.norm(v0_probe) > eps:
        v_s[0] = blend_first_step_pos * v0_probe + (1.0 - blend_first_step_pos) * v_s[0]

    pred_pos_s = np.empty_like(pred_pos)
    cur_p = p_last.copy()

    for i in range(Hp):
        cur_p = cur_p + v_s[i]
        pred_pos_s[i] = cur_p

    # Rotation smoothing
    q_last = probe_quat[-1]
    full_q = np.vstack([q_last[None, :], pred_quat])

    # Ensure quaternion continuity
    for i in range(1, len(full_q)):
        if np.dot(full_q[i-1], full_q[i]) < 0:
            full_q[i] = -full_q[i]

    omegas = np.empty((Hp, 3))

    for i in range(Hp):
        dq = quat_mul(quat_inv(full_q[i]), full_q[i+1])
        omegas[i] = quat_log(dq)

    omegas_s = moving_average_centered_orient(omegas, int(win))

    # Blend first rotation step
    dq_probe = quat_mul(quat_inv(probe_quat[-2]), probe_quat[-1])
    omega0_probe = quat_log(dq_probe)

    if np.linalg.norm(omega0_probe) > eps:
        omegas_s[0] = blend_first_step_rot * omega0_probe + (1.0 - blend_first_step_rot) * omegas_s[0]

    pred_quat_s = np.empty_like(pred_quat)
    cur_q = q_last.copy()

    for i in range(Hp):
        dq = quat_exp(omegas_s[i])

        # Shortest representation
        if dq[0] < 0:
            dq = -dq

        cur_q = quat_mul(cur_q, dq)
        cur_q = quat_normalize(cur_q)

        pred_quat_s[i] = cur_q

    return pred_pos_s, pred_quat_s

# ============================================================
# Save helpers
# ============================================================

def process_csv(input_path, output_path, freq=20, downsample=5):
    """
    Process raw CSV file by adding time column and downsampling.

    Args:
    - input_path: path to raw CSV file with columns [time, x, y, z, qx, qy, qz, qw]
    - output_path: path to save processed CSV
    - freq: frequency of the trajectory (for generating time column if not present)
    - downsample: int, if > 1, take every N-th row for downsampling
    """
    df = pd.read_csv(input_path)

    if downsample > 1:
        df = df.iloc[::downsample].reset_index(drop=True)

    dt = 1.0 / freq
    df["time"] = (np.arange(len(df)) * dt).round(2)

    df.to_csv(output_path, index=False)

    print(f"[Process CSV] Processed CSV saved to {output_path}")

def save_predictions_to_csv(
    filepath,
    preds,
    preds_quat,
    *,
    dt=0.005,
):
    """
    Save GP predicted trajectory (position + quaternion) to CSV.

    Args:
        filepath: output CSV file path
        preds: (N, 3) predicted positions
        preds_quat: (N, 4) predicted orientations as quaternions [w, x, y, z]
        dt: time step between predictions (for generating timestamps)
    """
    if preds is None or preds_quat is None:
        raise ValueError("preds or preds_quat is None")

    P = np.asarray(preds, dtype=np.float64)
    Q = np.asarray(preds_quat, dtype=np.float64)

    assert len(P) == len(Q), f"Length mismatch: preds has {len(P)} points, preds_quat has {len(Q)} points"

    N = len(P)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "time",
            "x", "y", "z",
            "qx", "qy", "qz", "qw"
        ])

        for i in range(N):
            t = i * dt

            x, y, z = P[i]
            qw, qx, qy, qz = Q[i]

            writer.writerow([
                round(float(t), 4),
                round(float(x), 6), round(float(y), 6), round(float(z), 6),
                round(float(qx), 6), round(float(qy), 6), round(float(qz), 6), round(float(qw), 6)
            ])

    print(f"[Save] Predictions saved to {filepath}")
