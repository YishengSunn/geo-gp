import numpy as np


def estimate_rotation_scale_3d(ref_pts, probe_pts, eps=1e-12):
    """
    Estimate similarity transform (rotation R, isotropic scale s, translation t)
    that best aligns ref_pts -> probe_pts in least-squares sense.

    Finds R in SO(3), s > 0, t in R^3 minimizing:
        sum_i || probe_i - (s * (ref_i @ R.T) + t) ||^2

    Args:
        ref_pts:  array-like (N, 3)
        probe_pts: array-like (N, 3)
        eps: numerical stability

    Returns:
        R: (3, 3) np.ndarray, rotation matrix
        s: float, isotropic scale
        t: (3,) np.ndarray, translation
    """
    X = np.asarray(ref_pts, dtype=np.float64)
    Y = np.asarray(probe_pts, dtype=np.float64)

    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != 3 or Y.shape[1] != 3:
        raise ValueError(f"Expected (N,3) inputs, got {X.shape} and {Y.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"N mismatch: {X.shape[0]} vs {Y.shape[0]}")
    if X.shape[0] < 3:
        raise ValueError("Need at least 3 point correspondences in 3D")

    # 1) Center (remove translation)
    mx = X.mean(axis=0)
    my = Y.mean(axis=0)
    Xc = X - mx
    Yc = Y - my

    # 2) Rotation via Kabsch/Procrustes (SVD)
    # We want: Yc ≈ s * Xc @ R.T  => (Xc^T Yc) relates the frames
    H = Xc.T @ Yc  # (3,3)
    U, Svals, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix reflection to ensure det(R)=+1 (proper rotation)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 3) Isotropic scale (least-squares optimal given R)
    # Min over s: ||Yc - s*(Xc @ R.T)||^2
    Xr = Xc @ R.T
    denom = np.sum(Xr * Xr) + eps
    s = float(np.sum(Yc * Xr) / denom)

    # (Optional) enforce positive scale (usually you want s>0)
    if s < 0:
        # Flip rotation 180° around any axis is messy; simplest is:
        # treat negative s as invalid and clamp or abs.
        # Here we take abs and keep R (common in practice).
        s = abs(s)

    # 4) Translation to match centroids
    t = my - s * (mx @ R.T)

    return R, s, t

def frame_from_window_transport(pts_hist_np, up=(0,0,1), eps=1e-12):
    """
    Build local frame from history points using parallel transport.

    Args:
        pts_hist_np: array-like (m, 3) of recent positions (need at least 2)
        up: fallback reference up vector
        eps: numerical stability

    Returns:
        R_end: (3, 3) np.ndarray, columns are [t, n, b] at the end of the window
        prev_b: (3,) np.ndarray, last binormal (for next step)
    """
    P = np.asarray(pts_hist_np, dtype=np.float64)
    prev_b = None
    R_end = None
    for t in range(1, len(P)):
        R_end, prev_b = local_frame_3d_from_points(P[:t+1], prev_b=prev_b, up=up, eps=eps)
    return R_end, prev_b

def local_frame_3d_from_points(
    pts,
    *,
    prev_b=None,
    up=(0.0, 0.0, 1.0),
    eps: float = 1e-12,
):
    """
    Build a smooth 3D local frame (right-handed) from recent points.

    This is a *stable* "tangent + transported binormal" frame meant for rollout:
    - t: tangent direction from the last segment
    - n: chosen to be orthogonal to t and as consistent as possible with prev_b
    - b: completes right-handed basis

    The continuity trick:
    - If prev_b is given, we compute n ~ prev_b x t.
    - If this degenerates (prev_b parallel to t), we fall back to a fixed 'up'.
    - Finally, we flip (n,b) together if b disagrees with prev_b.

    Args:
        pts: array-like (m, 3) of recent positions (need at least 2)
        prev_b: previous binormal (3,) or None
        up: fallback reference up vector
        eps: numerical stability

    Returns:
        R: (3, 3) np.ndarray, columns are [t, n, b]
        b: (3,) np.ndarray, current binormal (for next step)
    """
    P = np.asarray(pts, dtype=np.float64)

    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"pts must have shape (m,3), got {P.shape}")
    if P.shape[0] < 2:
        # Default frame
        t = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        n = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        b = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        R = np.stack([t, n, b], axis=1)
        return R, b

    # 1) Tangent (use last segment)
    t = np_normalize(P[-1] - P[-2], eps=eps)
    if float(np.linalg.norm(t)) < eps:
        # If last segment is degenerate, search backwards for a non-degenerate segment
        for i in range(P.shape[0] - 2, 0, -1):
            t = np_normalize(P[i] - P[i - 1], eps=eps)
            if float(np.linalg.norm(t)) >= eps:
                break
        if float(np.linalg.norm(t)) < eps:
            # Still degenerate
            t = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    up_v = np_normalize(np.asarray(up, dtype=np.float64), eps=eps)
    if float(np.linalg.norm(up_v)) < eps:
        up_v = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # 2) Choose n using prev_b if possible
    if prev_b is not None:
        prev_b = np_normalize(np.asarray(prev_b, dtype=np.float64), eps=eps)
        n_raw = np.cross(prev_b, t)
        if float(np.linalg.norm(n_raw)) < eps:
            # Degenerate: prev_b parallel to t -> fall back to up
            n_raw = np.cross(up_v, t)
    else:
        n_raw = np.cross(up_v, t)

    if float(np.linalg.norm(n_raw)) < eps:
        # Still degenerate: choose a different fallback axis
        alt = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(alt, t))) > 0.9:
            alt = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        n_raw = np.cross(alt, t)

    n = np_normalize(n_raw, eps=eps)

    # 3) Binormal completes right-handed basis
    b = np_normalize(np.cross(t, n), eps=eps)

    # Re-orthogonalize n in case of numerical drift
    n = np_normalize(np.cross(b, t), eps=eps)

    # 4) Continuity fix: keep b aligned with prev_b (avoid 180° flips)
    if prev_b is not None:
        if float(np.dot(b, prev_b)) < 0.0:
            b = -b
            n = -n

    R = np.stack([t, n, b], axis=1)  # columns are basis vectors
    return R, b

def estimate_global_up_from_traj(traj_np: np.ndarray):
    """
    Estimate a global 'up' direction from the whole trajectory using PCA.

    Args:
        traj_np: (T, 3) numpy array

    Returns:
        up: (3,) numpy array, unit vector
    """
    assert traj_np.ndim == 2 and traj_np.shape[1] == 3

    # Center trajectory
    X = traj_np - traj_np.mean(axis=0, keepdims=True)

    # PCA via covariance
    C = X.T @ X / max(len(X) - 1, 1)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # Principal axis = largest variance direction
    axis = eigvecs[:, 0]

    # Normalize
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    return axis

def np_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a vector; if too small, return zeros.
    
    Args:
        v: array-like vector
        eps: float, threshold for small norm
    
    Returns:
        normalized vector
    """
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=np.float64)
    return v / n

def rotate_world_to_local_3d(v_world, R: np.ndarray) -> np.ndarray:
    """
    Convert world vectors to local coordinates given R whose columns are [t,n,b].

    Args:
        v_world: array-like vector in world coordinates
        R: (3, 3) np.ndarray, columns are [t, n, b]

    Returns:
        v_local: array-like vector in local coordinates
    """
    v = np.asarray(v_world, dtype=np.float64)
    return v @ R  # Because columns are basis vectors

def rotate_local_to_world_3d(v_local, R: np.ndarray) -> np.ndarray:
    """
    Convert local vectors back to world coordinates given R whose columns are [t,n,b].

    Args:
        v_local: array-like vector in local coordinates
        R: (3, 3) np.ndarray, columns are [t, n, b]

    Returns:
        v_world: array-like vector in world coordinates
    """
    v = np.asarray(v_local, dtype=np.float64)
    return v @ R.T
