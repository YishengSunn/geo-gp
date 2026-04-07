import numpy as np

from utils.quaternion import quat_slerp


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
        raise ValueError(f"Expected points_xyz shape (N, 3), got {pts.shape}")

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
    points_quat: np.ndarray,
    *,
    sample_hz: float,
    speed: float,
    points_force: np.ndarray | None = None,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample a 6D trajectory (position + orientation) to equal time intervals.

    Position is resampled along arc-length assuming constant speed.
    Orientation is interpolated using quaternion SLERP.
    Optional force (per-point, same frame as position) is linearly interpolated
    along the same arc-length parameter as position.

    Args:
        points_xyz: (N, 3) array of 3D positions
        points_quat: (N, 4) array of quaternions [w, x, y, z]
        sample_hz: sampling frequency (Hz)
        speed: assumed constant speed (units per second)
        points_force: optional (N, 3) force samples aligned with points_xyz
        eps: numerical stability threshold

    Returns:
        pos_eq: (M, 3) resampled positions
        quat_eq: (M, 4) resampled quaternions [w, x, y, z]
        force_eq: (M, 3) resampled forces if points_force was given; otherwise omitted
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    quats = np.asarray(points_quat, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points_xyz (N,3), got {pts.shape}")
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(f"Expected points_quat (N,4), got {quats.shape}")
    if pts.shape[0] != quats.shape[0]:
        raise ValueError("Position and quaternion arrays must match length")

    pf = None
    if points_force is not None:
        pf = np.asarray(points_force, dtype=np.float64)
        if pf.ndim != 2 or pf.shape[1] != 3 or pf.shape[0] != pts.shape[0]:
            raise ValueError(f"Expected points_force (N,3) with N={pts.shape[0]}, got {pf.shape}")

    if pts.shape[0] < 2:
        if pf is None:
            return pts.copy(), quats.copy()
        return pts.copy(), quats.copy(), pf.copy()

    # 1) Segment lengths
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)

    total_len = float(np.sum(seg_len))
    if total_len < eps:
        if pf is None:
            return pts[:1].copy(), quats[:1].copy()
        return pts[:1].copy(), quats[:1].copy(), pf[:1].copy()

    # 2) Time parameterization
    total_time = total_len / float(speed)
    dt = 1.0 / float(sample_hz)

    t_samples = np.arange(0.0, total_time + 0.5 * dt, dt)
    s_samples = (t_samples / total_time) * total_len

    # 3) Cumulative arc-length
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])

    # 4) Interpolation
    pos_out = []
    quat_out = []
    force_out = [] if pf is not None else None

    j = 0
    for s in s_samples:
        while j < len(seg_len) - 1 and s > cum_len[j+1]:
            j += 1

        ds = s - cum_len[j]

        if seg_len[j] < eps:
            r = 0.0
        else:
            r = ds / seg_len[j]

        # Position interpolation
        p = pts[j] + r * seg[j]
        pos_out.append(p)

        # Quaternion interpolation (SLERP)
        q0 = quats[j]
        q1 = quats[j+1]

        q_interp = quat_slerp(q0, q1, r)
        quat_out.append(q_interp)

        if pf is not None:
            f = pf[j] + r * (pf[j+1] - pf[j])
            force_out.append(f)

    pos_eq = np.asarray(pos_out, dtype=np.float64)
    quat_eq = np.asarray(quat_out, dtype=np.float64)
    if pf is None:
        return pos_eq, quat_eq
    return pos_eq, quat_eq, np.asarray(force_out, dtype=np.float64)

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

    # 1) Segment lengths
    seg = P[1:] - P[:-1]
    seg_len = np.linalg.norm(seg, axis=1)

    # 2) Cumulative arc-length
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])

    # 3) Total length
    total = float(cum[-1])
    if total < eps:
        return np.repeat(P[:1], M, axis=0)

    # 4) Target cumulative lengths (uniform in fraction)
    s_targets = np.linspace(0.0, total, M)

    # 5) Linear interpolation along arc-length
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
