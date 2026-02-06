import numpy as np


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
