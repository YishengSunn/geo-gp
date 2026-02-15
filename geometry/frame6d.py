import numpy as np

from geometry.resample import resample_by_arclen_fraction


def estimate_rotation_scale_3d(ref_pts, probe_pts, eps=1e-12):
    """
    Estimate similarity transform (rotation R, isotropic scale s, translation t)
    that best aligns ref_pts -> probe_pts in least-squares sense.

    Finds R in SO(3), s > 0, t in R^3 minimizing:
        sum_i || probe_i - (s * (ref_i @ R.T) + t) ||^2

    Args:
        ref_pts: array-like (N, 3)
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

    # Enforce positive scale
    if s < 0:
        s = abs(s)

    # 4) Translation to match centroids
    t = my - s * (mx @ R.T)

    Yhat = s * (X @ R.T) + t
    rmse = float(np.sqrt(np.mean(np.sum((Y - Yhat) ** 2, axis=1))))

    return R, s, t, rmse

def estimate_rotation_scale_3d_search_by_count(
    ref_eq: np.ndarray,
    probe_eq: np.ndarray,
    *,
    margin_pts: int = 20,
    step: int = 1,
):
    """
    Estimate similarity transform (R,s,t) aligning ref_eq -> probe_eq
    by searching over possible segment lengths in ref_eq.
    Search best ref prefix length around probe length (with scale range prior).

    Args:
        ref_eq: (M,3) array-like, reference 3D curve points
        probe_eq: (N,3) array-like, probe 3D curve points
        margin_pts: int, number of extra points to consider around expected length
        step: int, step size for searching end index in ref_eq
    
    Returns:
        R: (3,3) np.ndarray, rotation matrix
        s: float, isotropic scale
        t: (3,) np.ndarray, translation
        j_end: int, end index in ref_eq used for best alignment
        rmse: float, root mean square error after alignment
    """
    ref_eq = np.asarray(ref_eq, dtype=np.float64)
    probe_eq = np.asarray(probe_eq, dtype=np.float64)

    Np = probe_eq.shape[0]
    j_center = Np
    j_lo = max(3, j_center - margin_pts)
    j_hi = min(len(ref_eq), j_center + margin_pts)

    best = None
    for j_end in range(j_lo, j_hi + 1, step):
        X = ref_eq[:j_end]
        Y = probe_eq  # Use all probe points

        Xr = resample_by_arclen_fraction(X, Np)  # Resample to match probe point count
        R, s, t, rmse = estimate_rotation_scale_3d(Xr, Y)
        if best is None or rmse < best[0]:
            best = (rmse, R, s, t, j_end)

    rmse, R, s, t, j_end = best
    return R, s, t, j_end, rmse
