import numpy as np


def wrap_pi(a):
    """
    Wrap angle to (-π, π]
    """
    return ((a + np.pi) % (2 * np.pi)) - np.pi

def angle_diff(a, b):
    """
    Calculate the minimum difference between two angles, range (-π, π]
    """
    return wrap_pi(a - b)

def angle_diff_mod_pi(a, b):
    """
    Calculate the minimum difference between two angles, range (-π, π]
    """
    return ((a - b + np.pi) % (2 * np.pi)) - np.pi

def estimate_start_tangent(xy, k: int = 5) -> float:
    """
    Estimate the starting tangent angle φ0 from the first k chords.
    Initial heading φ0 by chord-average from the origin:
    mean( P[i] - P[0] ), i=1..k (clamped by available points).
    This matches the 'global_base_dir' used in training/rollout.

    Args:
        xy: list or numpy array of shape (N, 2)
        k: number of chords to use (default 5)

    Returns:
        φ0: float, angle in radians (-π, π]
    """
    P = np.asarray(xy, dtype=np.float64)

    if P.shape[0] < 2:
        return 0.0
    
    # Use up to k chords from the origin
    m = int(min(max(1, P.shape[0] - 1), max(1, k)))  # 1 <= m <= P.shape[0]-1
    dirs = P[1:m+1] - P[0]  # Shape: (m, 2)
    v = dirs.mean(axis=0)  # Average direction
    if not np.isfinite(v).all() or np.linalg.norm(v) < 1e-12:
        v = P[1] - P[0]

    return float(np.arctan2(v[1], v[0]))

def angles_relative_to_start_tangent(points, k_hist, min_r=1e-3):
    """
    Calculate angles of points relative to the start tangent estimated from the first k_hist points.

    Args:
        points: list or numpy array of shape (N, 2)
        k_hist: int, number of points to estimate start tangent
        min_r: float, minimum radius to consider valid angle
    
    Returns:
        th_rel: numpy array of shape (N,), relative angles
        mask: numpy array of shape (N,), boolean mask indicating valid angles
    """
    P = np.asarray(points, dtype=np.float64)

    if len(P) == 0:
        return np.array([]), np.zeros(0, dtype=bool)
    
    o = P[0]  # Origin
    phi0 = estimate_start_tangent(P, k=k_hist)  # Starting tangent angle

    v = P - o  # Vectors from origin
    r = np.linalg.norm(v, axis=1)  # Radii, shape (N,)
    th = np.arctan2(v[:,1], v[:,0])  # Angles, shape (N,)

    th_rel = wrap_pi(th - phi0)  # Relative angles, shape (N,)
    mask = (r > min_r)

    return th_rel, mask

def build_relative_angles(xy, *, k_hist, origin_idx=0, min_r=1e-6):
    """
    Build relative angles θ_rel with respect to the starting tangent at origin_idx.

    Args:
        xy: list or numpy array of shape (N, 2)
        origin_idx: int, index of the origin point to compute relative angles from
        min_r: float, minimum radius to consider for angle calculation

    Returns:
        out: numpy array of shape (N,), relative angles in radians (-π, π], NaN for points with r < min_r
    """
    P = np.asarray(xy, dtype=np.float64)
    N = len(P)
    if N == 0:
        return np.array([], dtype=np.float64)
    sub = P[origin_idx:]
    th_rel_sub, _ = angles_relative_to_start_tangent(sub, k_hist=k_hist, min_r=min_r)
    out = np.full(N, np.nan, dtype=np.float64)
    out[origin_idx:origin_idx+len(th_rel_sub)] = th_rel_sub
    
    return out

def angles_with_phi0(points, k_hist, min_r):
    """
    Calculate angles relative to start tangent and also return the start tangent angle φ0.

    Compatible wrapper:
        - If your angles_relative_to_start_tangent returns (angles, mask, phi0), use it directly;
        - If it only returns (angles, mask), calculate phi0 here.

    Args:
        points: list or numpy array of shape (N, 2)
        k_hist: int, number of points to estimate start tangent
        min_r: float, minimum radius to consider valid angle

    Returns:
        th_rel: numpy array of shape (N,), relative angles
        mask: numpy array of shape (N,), boolean mask indicating valid angles
        phi0: float, starting tangent angle in radians
    """
    out = angles_relative_to_start_tangent(points, k_hist=k_hist, min_r=min_r)

    if isinstance(out, tuple) and len(out) == 3:
        return out  # (angles, mask, phi0)
    elif isinstance(out, tuple) and len(out) == 2:
        angles, mask = out
        phi0 = estimate_start_tangent(points, k=k_hist)
        return angles, mask, phi0
    else:
        raise RuntimeError("angles_relative_to_start_tangent returned an unexpected format")
    
def last_window_rel_angles(points, W, min_r=1e-3):
    """
    Calculate relative angles for the last W points in the trajectory.
    
    Args:
        points: list or numpy array of shape (N, 2)
        W: int, window size
        min_r: float, minimum radius to consider valid angle

    Returns:
        th: numpy array of shape (W,), relative angles
        m: numpy array of shape (W,), mask indicating valid angles    
    """
    P = np.asarray(points, dtype=np.float64)

    if P.shape[0] < 2:
        return None, None
    
    W = int(max(2, min(W if W is not None else 10, P.shape[0])))  # 2 <= W <= P.shape[0]
    th, m = angles_relative_to_start_tangent(P, k_hist=W, min_r=min_r)

    end = len(th) - 1
    start = max(0, end - (W - 1))

    return th[start:end+1], m[start:end+1]  # Return last W elements

def compute_base_unit_vec(points, n_segments=10):
    """
    Compute the base unit vector from the average of the first n_segments segments.
    
    Args:
        points: list or numpy array of shape (N, 2)
        n_segments: int, number of segments to average

    Returns:
        unit_vec: numpy array of shape (2,), base unit vector
    """
    pts = np.asarray(points, dtype=np.float64)

    m = min(n_segments, pts.shape[0] - 1)
    if m < 1:
        return np.array([1.0, 0.0], dtype=np.float64)
    
    seg = np.diff(pts[:m+1], axis=0)  # First m segments
    n = np.linalg.norm(seg, axis=1, keepdims=True)  # Segment lengths
    n[n < 1e-12] = 1.0
    
    u = seg / n  # Normalize segments
    v = u.mean(axis=0)  # Average direction
    if np.linalg.norm(v) < 1e-12:
        v = seg[0]

    return v / max(np.linalg.norm(v), 1e-12)
    
def first_index_reach_threshold(
    angles,
    mask,
    target,
    *,
    inclusive: bool = True
) -> int:
    """
    Check intervals between adjacent valid points (i_prev -> i_curr): if either target or -target
    lies within the minimal signed rotation interval from angles[i_prev] to angles[i_curr],
    return i_prev. Otherwise, fallback to the previous index of the "closer" point.

    Args:
        angles: array-like of shape (N,), angle sequence in radians
        mask: array-like of shape (N,), boolean mask indicating valid angles
        target: float, target angle in radians
        inclusive: bool, whether to include endpoints when checking intervals

    Returns:
        idx: int, index of the first point reaching the target angle condition

    - Angle unit: radians
    - Use minimal signed difference in (-pi, pi] to handle wrapping
    """
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return 0
    if idxs.size == 1:
        return int(idxs[0])

    def diff(a, b):
        """
        Calculate minimal signed difference from angle b to angle a.
        """
        # Minimal signed difference in (-pi, pi]
        return angle_diff_mod_pi(a, b)

    def bracket_hit(a, b, t) -> bool:
        """
        Check if t is within the minimal rotation interval from a to b.
        
        Args:
            a: start angle
            b: end angle
            t: target angle
            
        Returns:
            hit: bool, whether t is within the interval
        """
        d  = diff(b, a)   # Minimal rotation from a to b
        dt = diff(t, a)   # Minimal rotation from a to t
        if d > 0:
            return (0.0 <= dt <= d) if inclusive else (0.0 < dt < d)
        elif d < 0:
            return (d <= dt <= 0.0) if inclusive else (d < dt < 0.0)
        else:
            # d == 0: a and b coincide, hit only if t == a (considering numerical tolerance)
            return inclusive and (abs(dt) <= 1e-12)

    ang = np.asarray(angles, dtype=float)
    t_pos = float(target)
    t_neg = -t_pos

    # Scan adjacent valid index pairs
    for j in range(1, len(idxs)):
        i_prev = idxs[j - 1]
        i_curr = idxs[j]
        a = float(ang[i_prev])
        b = float(ang[i_curr])

        if bracket_hit(a, b, t_pos) or bracket_hit(a, b, t_neg):
            return int(i_prev)

    # Fallback: choose the one closer to either +target or -target
    diffs_pos = np.array([abs(diff(float(ang[i]), t_pos)) for i in idxs])
    diffs_neg = np.array([abs(diff(float(ang[i]), t_neg)) for i in idxs])
    diffs = np.minimum(diffs_pos, diffs_neg)

    k = int(idxs[int(np.argmin(diffs))])
    pos = int(np.where(idxs == k)[0][0])
    return int(idxs[pos - 1]) if pos > 0 else k

def crossed_multi_in_angle_rel(theta_from, theta_to, anchor_angles):
    """
    Detect whether the angle changes from theta_from to theta_to cross any angles in anchor_angles.
    Supports only 1 or 2 anchor angles.

    Args:
        theta_from: float, starting angle in radians
        theta_to: float, ending angle in radians
        anchor_angles: list/tuple/np.ndarray of 1 or 2 angles in radians

    Returns:
        crossed: bool, whether any anchor angle is crossed
        crossed_count: int, number of anchor angles crossed
    """
    assert isinstance(anchor_angles, (list, tuple, np.ndarray))
    assert len(anchor_angles) in [1, 2], "Supports only 1 or 2 anchor_angles"

    d_total = angle_diff(theta_to, theta_from)  # Difference (signed, (-π,π])

    crossed_count = 0
    for a in anchor_angles:
        d_anchor = angle_diff(a, theta_from)
        if d_total > 0 and 0 < d_anchor <= d_total:
            crossed_count += 1
        elif d_total < 0 and d_total <= d_anchor < 0:
            crossed_count += 1

    return crossed_count > 0, crossed_count
