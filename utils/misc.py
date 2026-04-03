import csv
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.quaternion import (
    rotmat_to_quat, quat_mul, quat_inv, quat_normalize,
    quat_log, quat_exp, quat_slerp
)


# ============================================================
# Standardization helpers
# ============================================================

class Standardizer:
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> "Standardizer":
        """
        Fit standardizer to data

        Args:
            X: input data, torch.Tensor of shape (N, D_in)
            Y: output data, torch.Tensor of shape (N, D_out)
            
        Returns:
            self
        """
        self.X_mean = X.mean(0)                # Shape: (D_in,)
        self.X_std = X.std(0).clamp_min(1e-8)  # Shape: (D_in,)
        self.Y_mean = Y.mean(0)                # Shape: (D_out,)
        self.Y_std = Y.std(0).clamp_min(1e-8)  # Shape: (D_out,)

        return self

    def x_transform(self, X: torch.Tensor) -> torch.Tensor: return (X - self.X_mean) / self.X_std

    def y_transform(self, Y: torch.Tensor) -> torch.Tensor: return (Y - self.Y_mean) / self.Y_std

    def y_inverse_transform(self, Yn: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform standardized output

        Args:
            Yn: standardized output, torch.Tensor of shape (..., D_out)

        Returns:
            Y: original output, torch.Tensor of shape (..., D_out)
        """
        assert Yn.shape[-1] == self.Y_std.shape[0], f"Dimension mismatch: Yn.shape={Yn.shape}, std={self.Y_std.shape}"

        return Yn * self.Y_std + self.Y_mean

    # Compatible with old interface
    def y_inverse(self, Yn: torch.Tensor) -> torch.Tensor: return self.y_inverse_transform(Yn)

# ============================================================
# Tensor/Array conversion helpers
# ============================================================

def torch_to_np(x: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor to numpy array
    
    Args:
        x: torch.Tensor

    Returns:
        np.ndarray
    """
    return x.detach().cpu().numpy()

# ============================================================
# Smoothing helpers (post-processing)
# ============================================================

def moving_average_centered_pos(arr: np.ndarray, win: int) -> np.ndarray:
    """
    Centered moving average smoothing along time axis (axis=0).

    Args:
        arr: (N, d) np.ndarray
        win: int, window size (odd preferred)

    Returns:
        out: (N, d) np.ndarray, smoothed array
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

def moving_average_centered_orient(arr: np.ndarray, win: int) -> np.ndarray:
    """
    Centered moving average smoothing along time axis (axis=0).

    Args:
        arr: (N, d) np.ndarray
        win: int, window size (odd preferred)

    Returns:
        out: (N, d) np.ndarray, smoothed array
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

    # Padding along time axis
    kernel = np.ones(win, dtype=np.float64) / float(win)
    out = arr.copy()

    for d in range(D):
        smoothed = np.convolve(arr[:, d], kernel, mode='valid')
        out[pad:N-pad, d] = smoothed
        
    return out

def moving_average_centered_6d(arr: np.ndarray, win: int) -> np.ndarray:
    """
    Centered moving average smoothing along time axis (axis=0).

    Supports:
    - (N, d) Euclidean features (position, spherical, etc.)
    - (N, 4) quaternions via log-exp smoothing

    Args:
        arr: np.ndarray of shape (N, d)
        win: int, window size (odd preferred)

    Returns:
        out: np.ndarray of shape (N, d), smoothed array
    """
    arr = np.asarray(arr, dtype=np.float64)

    # Quaternion
    if arr.ndim == 2 and arr.shape[1] == 4:
        # Ensure sign continuity
        qs = arr.copy()
        for i in range(1, len(qs)):
            if np.dot(qs[i-1], qs[i]) < 0:
                qs[i] = -qs[i]

        qs_s = moving_average_centered_orient(qs, int(win))

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
        probe: (Np, D) np.ndarray, probe points in world/probe frame
        pred: (Hp, D) np.ndarray, predicted points in world/probe frame (future, starting AFTER probe end)
        win: int, centered moving-average window (odd preferred)
        blend_first_step: float, blend factor for the first predicted delta:
            v0_smooth = blend_first_step * v0_probe + (1-blend_first_step) * v0_pred

    Returns:
        pred_smooth: (Hp, D) np.ndarray, smoothed predicted points in world/probe frame
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
        probe_pos: (Np, 3) np.ndarray, probe positions up to current time
        probe_quat: (Np, 4) np.ndarray, probe rotations as quaternions up to current time
        pred_pos: (Hp, 3) np.ndarray, predicted future positions (starting after probe end)
        pred_quat: (Hp, 4) np.ndarray, predicted future rotations as quaternions (starting after probe end)
        win: int, centered moving-average window (odd preferred)
        blend_first_step_pos: float, blend factor for first Δp
        blend_first_step_rot: float, blend factor for first ω
        eps: float, numerical threshold

    Returns:
        pred_pos_s: (Hp, 3) np.ndarray, smoothed predicted positions
        pred_quat_s: (Hp, 4) np.ndarray, smoothed predicted rotations as quaternions (still normalized)
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

def process_csv(input_path: str, output_path: str, freq: int = 20, downsample: int = 5):
    """
    Process raw CSV file by adding time column and downsampling.

    Args:
    - input_path: str, path to raw CSV file with columns [time, x, y, z, qx, qy, qz, qw]
    - output_path: str, path to save processed CSV
    - freq: int, frequency of the trajectory (for generating time column if not present)
    - downsample: int, if > 1, take every N-th row for downsampling
    """
    df = pd.read_csv(input_path)

    if downsample > 1:
        df = df.iloc[::downsample].reset_index(drop=True)

    dt = 1.0 / freq
    df["time"] = (np.arange(len(df)) * dt).round(2)

    df.to_csv(output_path, index=False)

    print(f"[Process CSV] Processed CSV saved to {output_path}")

def save_reference_raw_to_csv(
    filepath: str,
    ref_pos: np.ndarray,
    ref_quat: np.ndarray,
    *,
    dt: float = 0.05,
):
    """
    Save raw reference trajectory (position + quaternion) to CSV.

    Args:
        filepath: str, output CSV file path
        ref_pos: (N,3) np.ndarray, reference positions
        ref_quat: (N,4) np.ndarray, reference orientations [w,x,y,z]
        dt: float, time step between points
    """
    if ref_pos is None:
        raise ValueError("ref_pos is None")

    P = np.asarray(ref_pos, dtype=np.float64)
    N = len(P)

    if ref_quat is None:
        Q = np.zeros((N,4))
        Q[:,0] = 1.0
    else:
        Q = np.asarray(ref_quat, dtype=np.float64)

    assert len(P) == len(Q), f"Length mismatch: ref_pos={len(P)}, ref_quat={len(Q)}"

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
                round(float(x), 6),
                round(float(y), 6),
                round(float(z), 6),
                round(float(qx), 6),
                round(float(qy), 6),
                round(float(qz), 6),
                round(float(qw), 6),
            ])

    print(f"[Save] Reference raw trajectory saved to {filepath}")

def save_predictions_to_csv(
    filepath: str,
    preds: np.ndarray,
    preds_quat: np.ndarray,
    *,
    dt: float = 0.005,
):
    """
    Save GP predicted trajectory (position + quaternion) to CSV.

    Args:
        filepath: str, output CSV file path
        preds: (N, 3) np.ndarray, predicted positions
        preds_quat: (N, 4) np.ndarray, predicted orientations as quaternions [w, x, y, z]
        dt: float, time step between predictions (for generating timestamps)
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

# ============================================================
# Plot helpers
# ============================================================

def piecewise_quat_slerp_path(ref_seg: np.ndarray, t_samples: np.ndarray) -> np.ndarray:
    """
    Piecewise SLERP along ref_seg keyframes in [w,x,y,z], parameter t in [0,1]
    from first to last quaternion.

    Args:
        ref_seg: (M, 4) np.ndarray, reference quaternions [w, x, y, z]
        t_samples: (H,) np.ndarray, time samples in [0,1]

    Returns:
        out: (H, 4) np.ndarray, interpolated quaternions [w, x, y, z]
    """
    ref_seg = np.asarray(ref_seg, dtype=np.float64)
    t_samples = np.asarray(t_samples, dtype=np.float64)

    M = ref_seg.shape[0]
    H = len(t_samples)
    if M < 2:
        raise ValueError("Reference segment too short")
    
    t_keys = np.linspace(0.0, 1.0, M)
    out = np.empty((H, 4), dtype=np.float64)
    for i in range(H):
        t = float(np.clip(t_samples[i], 0.0, 1.0))
        if t <= t_keys[0]:
            out[i] = quat_normalize(ref_seg[0])
        elif t >= t_keys[-1]:
            out[i] = quat_normalize(ref_seg[-1])
        else:
            j = int(np.searchsorted(t_keys, t, side="right") - 1)
            j = min(max(j, 0), M - 2)
            t0, t1 = t_keys[j], t_keys[j + 1]
            u = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            out[i] = quat_slerp(ref_seg[j], ref_seg[j + 1], u)
    return out

def plot_orientation_error(
    ref_quat: np.ndarray,
    preds_quat: np.ndarray,
    start_idx: int,
    R_ref_probe: np.ndarray,
) -> np.ndarray:
    """
    Plot orientation error trend between reference and predicted quaternions, 
    after SLERP alignment and removing initial offset.

    Args:
        ref_quat: (N, 4) np.ndarray, reference rotations as quaternions [w, x, y, z]
        preds_quat: (H, 4) np.ndarray, predicted rotations as quaternions [w, x, y, z]
        start_idx: int, index in reference where prediction starts (after probe end)
        R_ref_probe: (3, 3) np.ndarray, reference to probe frame rotation

    Returns:
        errors: (H,) np.ndarray, orientation error trend in degrees
    """
    ref_quat = np.asarray(ref_quat, dtype=np.float64)
    preds_quat = np.asarray(preds_quat, dtype=np.float64)

    H = len(preds_quat)
    q_ref_probe = rotmat_to_quat(R_ref_probe)

    ref_seg = ref_quat[start_idx:]

    if len(ref_seg) < 2:
        raise ValueError("Reference segment too short")

    pred_progress = np.linspace(0, 1, H)
    ref_interp = piecewise_quat_slerp_path(ref_seg, pred_progress)

    q_ref0 = ref_interp[0]
    q_pred0 = preds_quat[0]

    errors = []

    for i in range(H):
        dq_ref = quat_mul(quat_inv(q_ref0), ref_interp[i])
        dq_ref = quat_mul(quat_mul(quat_inv(q_ref_probe), dq_ref), q_ref_probe)
        dq_pred = quat_mul(quat_inv(q_pred0), preds_quat[i])

        q_err = quat_mul(quat_inv(dq_ref), dq_pred)

        w = np.clip(abs(q_err[0]), -1.0, 1.0)
        theta = 2.0 * np.arccos(w)

        errors.append(np.degrees(theta))

    errors = np.array(errors)

    # Plot
    plt.figure()
    plt.plot(errors)
    plt.xlabel("Step")
    plt.ylabel("Orientation Error (deg)")
    plt.title("Orientation Trend Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return errors
