import numpy as np
import torch


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

def smooth_prediction_by_velocity(
    probe_xy: np.ndarray,
    pred_xy: np.ndarray,
    *,
    win: int = 9,
    blend_first_step: float = 0.8,
) -> np.ndarray:
    """
    Smooth predicted trajectory by smoothing per-step velocity (delta) in world/probe frame.

    This keeps the first predicted point continuous w.r.t. the last probe point, and blends the
    first velocity to preserve the probe's exiting direction.

    Args:
        probe_xy: (Np, 2) probe points in world/probe frame
        pred_xy:  (Hp, 2) predicted points in world/probe frame (future, starting AFTER probe end)
        win: centered moving-average window (odd preferred)
        blend_first_step: blend factor for the first predicted delta:
            v0_smooth = blend_first_step * v0_probe + (1-blend_first_step) * v0_pred

    Returns:
        pred_xy_smooth: (Hp, 2) smoothed predicted points in world/probe frame
    """
    probe_xy = np.asarray(probe_xy, dtype=np.float64)
    pred_xy = np.asarray(pred_xy, dtype=np.float64)

    if pred_xy.size == 0:
        return pred_xy
    if probe_xy.shape[0] < 2:
        return pred_xy

    # Build a full polyline that includes the last probe point as the boundary
    p_last = probe_xy[-1]
    full = np.vstack([p_last[None, :], pred_xy])  # (Hp+1, 2)

    # Velocity sequence for the predicted part
    v = np.diff(full, axis=0)  # (Hp, 2)

    # Smooth velocities using a centered moving average
    v_s = moving_average_centered(v, int(win))

    # Enforce a smooth connection direction using the probe's last delta
    v0_probe = probe_xy[-1] - probe_xy[-2]
    if np.linalg.norm(v0_probe) > 1e-12:
        v_s[0] = float(blend_first_step) * v0_probe + (1.0 - float(blend_first_step)) * v_s[0]

    # Reconstruct smoothed prediction points
    out = np.empty_like(pred_xy, dtype=np.float64)
    cur = p_last.copy()
    for i in range(pred_xy.shape[0]):
        cur = cur + v_s[i]
        out[i] = cur

    return out
