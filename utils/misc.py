import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp


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
# Quaternion helpers
# ============================================================

def normalize_quaternion(q, eps=1e-12):
    """
    Normalize a quaternion to unit length, with numerical stability.

    Args:
        q: array-like of shape (4,) in [w, x, y, z] format
        eps: small value to prevent division by zero

    Returns:
        normalized quaternion of shape (4,) in [w, x, y, z] format
    """
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n

def enforce_quaternion_hemisphere(quats):
    """
    Enforce quaternions to be in the same hemisphere to ensure smooth interpolation.
    This is done by iterating through the sequence and flipping the sign of any quaternion
    that has a negative dot product with the previous one.

    Args:
        quats: (N, 4) array of quaternions in [w, x, y, z] format

    Returns:
        (N, 4) array of quaternions with consistent hemisphere
    """
    quats = np.asarray(quats, dtype=np.float64).copy()
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(f"Expected quaternion array of shape (N, 4), got {quats.shape}")

    if quats.shape[0] == 0:
        return quats

    quats[0] = normalize_quaternion(quats[0])
    for i in range(1, quats.shape[0]):
        quats[i] = normalize_quaternion(quats[i])
        if np.dot(quats[i-1], quats[i]) < 0.0:
            quats[i] = -quats[i]

    return quats

def rotation_matrices_to_quat_wxyz(rot_mats):
    """
    Convert rotation matrices to quaternions in [w, x, y, z] format.

    Args:
        rot_mats: (N, 3, 3) array of rotation matrices

    Returns:
        (N, 4) array of quaternions in [w, x, y, z] format
    """
    rot_mats = np.asarray(rot_mats, dtype=np.float64)
    q_xyzw = R.from_matrix(rot_mats).as_quat()
    q_wxyz = np.stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]], axis=1)
    return enforce_quaternion_hemisphere(q_wxyz)

def quat_wxyz_to_rotation_matrices(quats):
    """
    Convert quaternions in [w, x, y, z] format to rotation matrices.

    Args:
        quats: (N, 4) array of quaternions in [w, x, y, z] format

    Returns:
        (N, 3, 3) array of rotation matrices
    """
    quats = np.asarray(quats, dtype=np.float64)
    q_xyzw = np.stack([quats[:, 1], quats[:, 2], quats[:, 3], quats[:, 0]], axis=1)
    q_xyzw = np.array([normalize_quaternion(q) for q in q_xyzw])
    return R.from_quat(q_xyzw).as_matrix()

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

def moving_average_centered_quat(quats, win):
    """
    Centered moving average smoothing for quaternions, with hemisphere enforcement.

    Args:
        quats: (N, 4) array of quaternions in [w, x, y, z] format
        win: int, window size (odd preferred)

    Returns:
        (N, 4) array of smoothed quaternions in [w, x, y, z] format
    """
    quats = enforce_quaternion_hemisphere(quats)
    win = int(win)

    if win < 3 or quats.shape[0] <= 1:
        return quats.copy()
    if win % 2 == 0:
        win += 1

    pad = win // 2
    N = quats.shape[0]
    out = np.empty_like(quats)

    for i in range(N):
        lo = max(0, i - pad)
        hi = min(N, i + pad + 1)
        local = quats[lo:hi].copy()
        anchor = quats[i]

        for j in range(local.shape[0]):
            if np.dot(anchor, local[j]) < 0.0:
                local[j] = -local[j]

        q_avg = local.mean(axis=0)
        out[i] = normalize_quaternion(q_avg)

    return enforce_quaternion_hemisphere(out)

def moving_average_centered_6d(arr, win):
    """
    Centered moving average smoothing along time axis (axis=0).

    Supports:
        - (N, d) Euclidean features (position, spherical, etc.)
        - (N, 3, 3) rotation matrices (SO(3)) via quaternion smoothing

    Args:
        arr: array-like
        win: int, window size (odd preferred)

    Returns:
        out: smoothed array, same shape as input
    """
    arr = np.asarray(arr, dtype=np.float64)

    # Rotation
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        quats = rotation_matrices_to_quat_wxyz(arr)
        quats_s = moving_average_centered_quat(quats, win)
        R_s = quat_wxyz_to_rotation_matrices(quats_s)
        return R_s
    
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

    v_s = moving_average_centered_pos(v, int(win))

    # Blend first step with probe's last delta
    v0_probe = probe_pos[-1] - probe_pos[-2]
    if np.linalg.norm(v0_probe) > eps:
        v_s[0] = float(blend_first_step_pos) * v0_probe + (1.0 - float(blend_first_step_pos)) * v_s[0]

    pred_pos_s = np.empty_like(pred_pos)
    cur_p = p_last.copy()
    for i in range(Hp):
        cur_p = cur_p + v_s[i]
        pred_pos_s[i] = cur_p

    R_last = probe_rot[-1]
    full_R = np.concatenate([R_last[None, :, :], pred_rot], axis=0)  # (Hp+1, 3, 3)

    dR_seq = np.empty((Hp, 3, 3), dtype=np.float64)
    for i in range(Hp):
        dR_seq[i] = full_R[i].T @ full_R[i+1]

    dq_seq = rotation_matrices_to_quat_wxyz(dR_seq)
    dq_s = moving_average_centered_quat(dq_seq, int(win))

    # Blend first step with probe's last angular increment.
    dR_probe = probe_rot[-2].T @ probe_rot[-1]
    dq_probe = rotation_matrices_to_quat_wxyz(dR_probe[None, :, :])[0]
    if np.linalg.norm(dq_probe[1:]) > eps or abs(dq_probe[0] - 1.0) > eps:
        if np.dot(dq_probe, dq_s[0]) < 0.0:
            dq_probe = -dq_probe
        dq_s[0] = normalize_quaternion(
            float(blend_first_step_rot) * dq_probe
            + (1.0 - float(blend_first_step_rot)) * dq_s[0]
        )

    dR_s = quat_wxyz_to_rotation_matrices(dq_s)

    pred_rot_s = np.empty_like(pred_rot)
    cur_R = R_last.copy()
    for i in range(Hp):
        cur_R = cur_R @ dR_s[i]
        pred_rot_s[i] = cur_R

    return pred_pos_s, pred_rot_s

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

# ============================================================
# Plot helpers
# ============================================================

def plot_orientation_error(ref_rot, pred_rot, start_idx, R_ref_probe):
    """
    Plot orientation error (in degrees) between predicted and reference rotations, 
    after aligning the first predicted pose.

    Args:
        ref_rot: (N, 3, 3) reference rotations
        pred_rot: (H, 3, 3) predicted rotations
        start_idx: int, index in reference where prediction starts (after probe end)
    """
    ref_rot = np.asarray(ref_rot, dtype=np.float64)
    pred_rot = np.asarray(pred_rot, dtype=np.float64)

    H = len(pred_rot)
    ref_seg = ref_rot[start_idx:]

    if len(ref_seg) < 2:
        raise ValueError("Reference segment too short")

    # SLERP progress alignment
    ref_R = R.from_matrix(ref_seg)

    ref_progress = np.linspace(0, 1, len(ref_seg))
    pred_progress = np.linspace(0, 1, H)

    slerp = Slerp(ref_progress, ref_R)
    ref_interp = slerp(pred_progress).as_matrix()

    # Remove initial orientation offset
    R_ref0 = ref_interp[0]
    R_pred0 = pred_rot[0]

    errors = []

    for i in range(H):
        dR_ref = R_ref0.T @ ref_interp[i]
        dR_ref = R_ref_probe.T @ dR_ref @ R_ref_probe
        dR_pred = R_pred0.T @ pred_rot[i]

        R_err = dR_ref.T @ dR_pred

        cos_theta = (np.trace(R_err) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        theta = np.arccos(cos_theta)
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
