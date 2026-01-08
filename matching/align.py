import numpy as np

from config.runtime import PHI0_K_REF, PHI0_K_PROBE
from geometry.angles import (
    estimate_start_tangent,
    angles_relative_to_start_tangent,
    angles_with_phi0,
    compute_base_unit_vec,
    first_index_reach_threshold
)


def estimate_similarity_by_vectors_only(anchor_pairs):
    """
    Given several anchor point pairs (pt_ref, pt_probe), estimate the overall rotation angle dtheta and scale.
    Does not require time information t_ref / t_probe, only uses vectors.

    Args:
        anchor_pairs: list of dicts, each with keys:
            'pt_ref': (x,y) of reference point
            'pt_probe': (x,y) of probe point
            Optional: 'ref_start': (x,y) of reference start point
                      'probe_start': (x,y) of probe start point
    
    Returns:
        dtheta: float, estimated rotation angle in radians
        scale: float, estimated scale factor
        n_used: int, number of anchor pairs used for estimation
    """
    v_refs = []
    v_probes = []

    for pair in anchor_pairs:
        pt_ref = np.asarray(pair['pt_ref'], dtype=np.float64)
        pt_probe = np.asarray(pair['pt_probe'], dtype=np.float64)
        if 'ref_start' in pair:
            ref_start = np.asarray(pair['ref_start'], dtype=np.float64)
        else:
            ref_start = np.zeros(2)  # Default start point is (0,0)
        if 'probe_start' in pair:
            probe_start = np.asarray(pair['probe_start'], dtype=np.float64)
        else:
            probe_start = np.zeros(2)

        v_ref = pt_ref - ref_start
        v_probe = pt_probe - probe_start

        # Exclude vectors that are too short to avoid numerical instability
        if np.linalg.norm(v_ref) < 1e-3 or np.linalg.norm(v_probe) < 1e-3:
            continue

        v_refs.append(v_ref)
        v_probes.append(v_probe)

    if len(v_refs) < 1:
        return None, None, 0  # Insufficient for estimation

    v_refs = np.stack(v_refs, axis=0)
    v_probes = np.stack(v_probes, axis=0)

    # === Calculate Δθ (average angle difference)
    def angle_between(v1, v2):
        """
        Calculate angle from v1 to v2
        """
        return np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])

    dthetas = [angle_between(vr, vp) for vr, vp in zip(v_refs, v_probes)]
    dtheta = np.mean(dthetas)

    # Calculate scale (average length ratio)
    norms_ref = np.linalg.norm(v_refs, axis=1)
    norms_probe = np.linalg.norm(v_probes, axis=1)
    scales = norms_probe / norms_ref
    scale = np.mean(scales)

    return dtheta, scale, len(v_refs)

def get_anchor_correspondence(ref_pts, probe_pts, angle_target, *, min_r: float = 1e-3, n_segments_base: int = 10):
    """
    Non-plotting version: Given raw reference/probe trajectories, find the first index where the relative starting tangent angle = ±angle_target is crossed,
    and return the vectors from origin to anchor points and the base unit vectors (average direction of the first n segments) for both trajectories.

    Args:
        ref_pts: (N,2) numpy array, reference trajectory
        probe_pts: (M,2) numpy array, probe trajectory
        angle_target: target relative angle (rad)
        min_r: float, minimum radius to consider valid angle
        n_segments_base: int, number of initial segments used for the base vector (default 10)

    Returns:
        dict with keys:
            "ref_index": index in ref_pts where angle_target is reached
            "ref_vector": vector from origin to matched point in ref_pts
            "ref_point": matched point in ref_pts
            "ref_base_unit": base unit vector of ref_pts
            "probe_index": index in probe_pts where angle_target is reached
            "probe_vector": vector from origin to matched point in probe_pts
            "probe_point": matched point in probe_pts
            "probe_base_unit": base unit vector of probe_pts
    """
    ref_pts = np.asarray(ref_pts, dtype=np.float64)
    probe_pts = np.asarray(probe_pts, dtype=np.float64)
    assert ref_pts.shape[0] >= 2 and probe_pts.shape[0] >= 2, "Ref/probe require at least 2 points"

    # Keep consistent with your existing angle functions
    def angles_with_phi0_local(points, k_hist, min_r):
        """
        Calculate angles relative to starting tangent, returning angles, mask, and starting tangent angle φ0.

        Args:
            points: list or numpy array of shape (N, 2)
            k_hist: int, number of points to estimate start tangent
            min_r: float, minimum radius to consider valid angle

        Returns:
            angles: numpy array of shape (N,), relative angles
            mask: numpy array of shape (N,), boolean mask indicating valid angles
            phi0: float, starting tangent angle in radians
        """
        out = angles_relative_to_start_tangent(points, k_hist=k_hist, min_r=min_r)

        if isinstance(out, tuple) and len(out) == 3:
            return out
        elif isinstance(out, tuple) and len(out) == 2:
            angles, mask = out
            phi0 = estimate_start_tangent(points, k=k_hist)
            return angles, mask, phi0
        else:
            raise RuntimeError("angles_relative_to_start_tangent returned an unexpected format")

    ref_ang, ref_mask, _ = angles_with_phi0_local(ref_pts, k_hist=PHI0_K_REF, min_r=min_r)
    pro_ang, pro_mask, _ = angles_with_phi0_local(probe_pts, k_hist=PHI0_K_PROBE, min_r=min_r)

    i_ref = first_index_reach_threshold(ref_ang, ref_mask, angle_target, inclusive=True)
    i_pro = first_index_reach_threshold(pro_ang, pro_mask, angle_target, inclusive=True)

    o_ref, p_ref = ref_pts[0], ref_pts[i_ref]  # Origin and matched point
    o_pro, p_pro = probe_pts[0], probe_pts[i_pro]  # Origin and matched point
    v_ref = p_ref - o_ref
    v_pro = p_pro - o_pro

    def base_(points, n_segments):
        """
        Compute base unit vector (average direction of first n_segments segments)
        
        Args:
            points: (N,2) numpy array
            n_segments: int, number of segments to consider
            
        Returns:
            unit vector: (2,) numpy array
        """
        m = min(n_segments, points.shape[0]-1)
        if m < 1:
            return np.array([1.0, 0.0], dtype=np.float64)
        seg = np.diff(points[:m+1], axis=0)
        n = np.linalg.norm(seg, axis=1, keepdims=True); n[n < 1e-12] = 1.0
        u = seg / n
        v = u.mean(axis=0)
        if np.linalg.norm(v) < 1e-12:
            v = seg[0]

        return v / max(np.linalg.norm(v), 1e-12)

    u_ref = base_(ref_pts, n_segments_base)
    u_pro = base_(probe_pts, n_segments_base)

    return {
        "ref_index": i_ref,
        "ref_vector": v_ref,
        "ref_point": p_ref,
        "ref_base_unit": u_ref,
        "probe_index": i_pro,
        "probe_vector": v_pro,
        "probe_point": p_pro,
        "probe_base_unit": u_pro,
    }

def plot_vectors_at_angle_ref_probe(
    ref_pts, probe_pts, angle_target, *,
    k_hist=10, min_r=1e-3, n_segments_base=10
):
    """
    For both reference and probe trajectories:
    1. Calculate relative angles to starting tangent.
    2. Find the first index where the angle reaches the target (or -target).
    3. Compute the target vector and base tangent vector.
    4. Plot the trajectories with target and base vectors.

    Args:
        ref_pts: list or numpy array of shape (N, 2)
        probe_pts: list or numpy array of shape (M, 2)
        angle_target: float, target angle in radians
        k_hist: int, number of points to estimate start tangent
        min_r: float, minimum radius to consider valid angle
        n_segments_base: int, number of segments to average for base tangent
    
    Returns:
        dict with keys:
            'ref_index', 'ref_vector', 'ref_point', 'ref_base_unit',
            'probe_index', 'probe_vector', 'probe_point', 'probe_base_unit'

    In two subplots of the same figure, plot:
    - Reference trajectory and probe's target-angle corresponding vector (origin -> matched point)
    - Reference trajectory and probe's "base vector" (average direction of the first n_segments_base segments)

    ref_pts, probe_pts : (N,2) / (M,2)
    angle_target       : target relative angle (rad)
    k_hist             : window for estimating the starting tangent in the angle curve (aligned with your project's K_HIST)
    n_segments_base    : number of initial segments used for the base vector (consistent with your algorithm, default 10)
    """
    print(f"Plotting target vectors at angle_target = {angle_target:.2f} rad ({np.degrees(angle_target):.1f}°)...")

    ref_pts   = np.asarray(ref_pts,   dtype=np.float64)
    probe_pts = np.asarray(probe_pts, dtype=np.float64)
    assert ref_pts.shape[0] >= 2 and probe_pts.shape[0] >= 2, "ref/probe 点数至少为2"

    # Angle curves + base angles
    ref_ang, ref_mask, _  = angles_with_phi0(ref_pts,  k_hist=k_hist, min_r=min_r)
    pro_ang, pro_mask, _  = angles_with_phi0(probe_pts, k_hist=k_hist, min_r=min_r)

    def masked_nearest_idx(angles, mask, target):
        """
        Find the index of the angle closest to target among valid points.

        Args:
            angles: array-like of shape (N,), angle sequence in radians
            mask: array-like of shape (N,), boolean mask indicating valid angles
            target: float, target angle in radians

        Returns:
            idx: int, index of the closest valid angle
        """
        if not np.any(mask):
            return 0
        idxs = np.where(mask)[0]
        return int(idxs[np.argmin(np.abs(angles[idxs] - target))])

    i_ref = first_index_reach_threshold(ref_ang, ref_mask, angle_target, inclusive=True)
    print("i_ref (by threshold) =", i_ref, "; nearest =", masked_nearest_idx(ref_ang, ref_mask, angle_target))
    i_pro = first_index_reach_threshold(pro_ang, pro_mask, angle_target, inclusive=True)
    print("i_pro (by threshold) =", i_pro, "; nearest =", masked_nearest_idx(pro_ang, pro_mask, angle_target))

    # Origin and target vectors
    o_ref, p_ref = ref_pts[0], ref_pts[i_ref]
    o_pro, p_pro = probe_pts[0], probe_pts[i_pro]
    v_ref = p_ref - o_ref
    v_pro = p_pro - o_pro

    # Base unit vectors (strictly average direction of the first 10 segments)
    u_ref = compute_base_unit_vec(ref_pts, n_segments=n_segments_base)
    u_pro = compute_base_unit_vec(probe_pts, n_segments=n_segments_base)

    # Give the base vectors a suitable display length (only affects visualization, does not change direction)
    L_ref = max(np.linalg.norm(v_ref), 1e-6) * 0.6
    L_pro = max(np.linalg.norm(v_pro), 1e-6) * 0.6
    b_ref = u_ref * L_ref
    b_pro = u_pro * L_pro

    """
    # ---- 绘图 ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    # 左: Reference
    axL.plot(ref_pts[:,0], ref_pts[:,1], '-', label='Reference Traj')
    axL.scatter(o_ref[0], o_ref[1], c='k', s=30, label='Origin')
    axL.scatter(p_ref[0], p_ref[1], c='g', s=60, marker='x', label=f'Target idx={i_ref}')
    # Target 向量
    axL.plot([o_ref[0], o_ref[0] + v_ref[0]], [o_ref[1], o_ref[1] + v_ref[1]],
            linewidth=2, color='g', label='Target Vector')
    # 基准向量 (前 10 段平均方向)
    axL.plot([o_ref[0], o_ref[0] + b_ref[0]], [o_ref[1], o_ref[1] + b_ref[1]],
            linestyle='--', linewidth=2, color='r', label='Base Tangent')
    axL.set_aspect('equal', adjustable='box')
    axL.grid(True, alpha=0.3)
    axL.set_title(f"Reference | Target={angle_target:.2f} rad ({np.degrees(angle_target):.1f}°)")
    axL.legend(loc='best', fontsize=9)

    # 右: Probe
    axR.plot(probe_pts[:,0], probe_pts[:,1], '-', label='Probe Traj')
    axR.scatter(o_pro[0], o_pro[1], c='k', s=30, label='Origin')
    axR.scatter(p_pro[0], p_pro[1], c='g', s=60, marker='x', label=f'Target idx={i_pro}')
    # Target 向量
    axR.plot([o_pro[0], o_pro[0] + v_pro[0]], [o_pro[1], o_pro[1] + v_pro[1]],
            linewidth=2, color='g', label='Target Vector')
    # 基准向量 (前 10 段平均方向)
    axR.plot([o_pro[0], o_pro[0] + b_pro[0]], [o_pro[1], o_pro[1] + b_pro[1]],
            linestyle='--', linewidth=2, color='r', label='Base tangent')
    axR.set_aspect('equal', adjustable='box')
    axR.grid(True, alpha=0.3)
    axR.set_title("Probe (Same Target Definition)")
    axR.legend(loc='best', fontsize=9)

    plt.suptitle("Target Vector & Base Tangent (Reference vs Probe)")
    plt.tight_layout()
    plt.show()
    """

    print("Plotting completed...")

    return {
        "ref_index": i_ref,   "ref_vector": v_ref,   "ref_point": p_ref,   "ref_base_unit": u_ref,
        "probe_index": i_pro, "probe_vector": v_pro, "probe_point": p_pro, "probe_base_unit": u_pro
    }
