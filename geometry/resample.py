import numpy as np

from geometry.so3 import so3_log, so3_exp


def resample_polyline_equal_dt(points_xy, sample_hz, speed):
    """
    Resample polyline points to have equal time intervals based on speed and sample_hz.
    
    Args:
        points_xy: list or numpy array of shape (N, 2)
        sample_hz: int, sampling frequency (samples per second)
        speed: float, speed (units per second)
        
    Returns:
        resampled_points: numpy array of shape (M, 2)
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.shape[0] < 2:
        return pts
    
    seg = pts[1:] - pts[:-1]  # Segments
    seg_len = np.linalg.norm(seg, axis=1)  # Segment lengths

    L = float(np.sum(seg_len))  # Total length
    if L <= 1e-8:
        return pts[:1]
    
    T_total = L / float(speed)
    dt = 1.0 / float(sample_hz)
    t_samples = np.arange(0.0, T_total+1e-9, dt)  # Sample times
    s_samples = (t_samples / T_total) * L  # Corresponding arc lengths
    cum_s = np.concatenate([[0.0], np.cumsum(seg_len)])  # Cumulative lengths

    out = []; j = 0
    for s in s_samples:
        # cum_s[j] <= s <= cum_s[j+1]
        while j < len(seg_len) - 1 and s > cum_s[j+1]:
            j += 1

        ds = s - cum_s[j]
        r = 0.0 if seg_len[j] < 1e-9 else ds / seg_len[j]
        p = pts[j] + r * seg[j]
        out.append(p)

    return np.asarray(out, dtype=np.float32)

def resample_trajectory_3d_equal_dt(
    points_xyz: np.ndarray,
    *,
    sample_hz: float,
    speed: float,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Resample a 3D trajectory to equal time intervals assuming constant speed.

    Args:
        points_xyz: (N, 3) array of 3D points
        sample_hz: sampling frequency (Hz)
        speed: assumed constant speed (units per second)
        eps: numerical stability threshold

    Returns:
        traj_eq: (M, 3) array of resampled 3D points
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points_xyz shape (N, 3), got {pts.shape}!")

    if pts.shape[0] < 2:
        return pts.copy()

    # 1) Segment vectors and lengths
    seg = pts[1:] - pts[:-1]               # (N-1, 3)
    seg_len = np.linalg.norm(seg, axis=1)  # (N-1,)

    total_len = float(np.sum(seg_len))
    if total_len < eps:
        return pts[:1].copy()

    # 2) Time parameterization
    total_time = total_len / float(speed)
    dt = 1.0 / float(sample_hz)

    t_samples = np.arange(0.0, total_time + 0.5 * dt, dt)
    s_samples = (t_samples / total_time) * total_len

    # 3) Cumulative arc-length
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])

    # 4) Linear interpolation along arc-length
    out = []
    j = 0
    for s in s_samples:
        while j < len(seg_len) - 1 and s > cum_len[j+1]:
            j += 1

        ds = s - cum_len[j]
        if seg_len[j] < eps:
            p = pts[j]
        else:
            r = ds / seg_len[j]
            p = pts[j] + r * seg[j]

        out.append(p)

    return np.asarray(out, dtype=np.float64)

def resample_trajectory_6d_equal_dt(
    points_xyz: np.ndarray,
    points_rot: np.ndarray,
    *,
    sample_hz: float,
    speed: float,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample a 6D trajectory (position + orientation) to equal time intervals.
    
    Position is resampled along arc-length assuming constant speed.
    Orientation is interpolated using SO(3) logarithm/exponential.

    Args:
        points_xyz: (N, 3) array of 3D positions
        points_rot: (N, 3, 3) array of rotation matrices
        sample_hz: sampling frequency (Hz)
        speed: assumed constant speed (units per second)
        eps: numerical stability threshold

    Returns:
        pos_eq: (M, 3) resampled positions
        rot_eq: (M, 3, 3) resampled orientations
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    rots = np.asarray(points_rot, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points_xyz shape (N,3), got {pts.shape}")
    if rots.ndim != 3 or rots.shape[1:] != (3, 3):
        raise ValueError(f"Expected points_rot shape (N,3,3), got {rots.shape}")
    if pts.shape[0] != rots.shape[0]:
        raise ValueError("Position and rotation arrays must have same length!")

    if pts.shape[0] < 2:
        return pts.copy(), rots.copy()

    # 1) Segment vectors and lengths
    seg = pts[1:] - pts[:-1]               # (N-1, 3)
    seg_len = np.linalg.norm(seg, axis=1)  # (N-1,)

    total_len = float(np.sum(seg_len))
    if total_len < eps:
        return pts[:1].copy(), rots[:1].copy()

    # 2) Time parameterization
    total_time = total_len / float(speed)
    dt = 1.0 / float(sample_hz)

    t_samples = np.arange(0.0, total_time + 0.5 * dt, dt)
    s_samples = (t_samples / total_time) * total_len

    # 3) Cumulative arc-length
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])

    # 4) Linear interpolation along arc-length
    pos_out = []
    rot_out = []

    j = 0
    for s in s_samples:
        while j < len(seg_len) - 1 and s > cum_len[j+1]:
            j += 1

        ds = s - cum_len[j]
        if seg_len[j] < eps:
            r = 0.0
        else:
            r = ds / seg_len[j]

        p = pts[j] + r * seg[j]
        pos_out.append(p)

        # Orientation interpolation on SO(3)
        R0 = rots[j]
        R1 = rots[j+1]
        dR = R0.T @ R1
        omega = so3_log(dR)
        R_interp = R0 @ so3_exp(r * omega)
        rot_out.append(R_interp)

    return np.asarray(pos_out, dtype=np.float64), np.asarray(rot_out, dtype=np.float64)

def resample_by_arclen_fraction(P: np.ndarray, M: int, eps: float = 1e-9) -> np.ndarray:
    """
    Resample polyline P (N,3) into M points uniformly in arclength fraction [0,1].

    Args:
        P: (N,3) array of 3D points
        M: int, number of points to resample to
        eps: float, numerical stability threshold

    Returns:
        out: (M,3) array of resampled 3D points
    """
    P = np.asarray(P, dtype=np.float64)
    if P.shape[0] < 2:
        return np.repeat(P[:1], M, axis=0)

    seg = P[1:] - P[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total < eps:
        return np.repeat(P[:1], M, axis=0)

    # Target cumulative lengths (uniform in fraction)
    s_targets = np.linspace(0.0, total, M)

    out = np.empty((M, 3), dtype=np.float64)
    j = 0
    for i, s in enumerate(s_targets):
        while j < len(seg_len) - 1 and s > cum[j+1]:
            j += 1
        ds = s - cum[j]
        if seg_len[j] < eps:
            out[i] = P[j]
        else:
            r = ds / seg_len[j]
            out[i] = P[j] + r * seg[j]
    return out

def resample_to_k(points_xy, k):
    """
    Resample polyline points to have exactly k points evenly spaced along the path.

    Args:
        points_xy: list or numpy array of shape (N, 2)
        k: int, number of points to resample to

    Returns:
        resampled_points: numpy array of shape (k, 2)
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape[0] < 2:
        return np.repeat(pts[:1], k, axis=0) if pts.size else np.zeros((k,2), dtype=np.float64)
    
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)  # Shape: (N-1,)

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    L = cum[-1]
    if L < 1e-9:
        return np.tile(pts[:1], (k,1))
    
    s = np.linspace(0.0, L, k)
    out = []; j = 0
    for si in s:
        # cum[j] <= si <= cum[j+1]
        while j < len(seg_len) - 1 and si > cum[j+1]:
            j += 1

        ds = si - cum[j]
        r = 0.0 if seg_len[j] < 1e-9 else ds/seg_len[j]

        p = pts[j] + r * seg[j]
        out.append(p)

    return np.asarray(out, dtype=np.float64)
