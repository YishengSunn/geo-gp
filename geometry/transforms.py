import numpy as np
import torch

from geometry.angles import angles_relative_to_start_tangent, first_index_reach_threshold
from geometry.resample import resample_to_k
from config.runtime import K_HIST


def rotate_to_fixed_frame(vectors, base_dir):
    """
    Rotate vectors to a fixed frame defined by base_dir
    Args:
        vectors: torch tensor of shape (N, 2)
        base_dir: torch tensor of shape (2,)
    
    Returns:
        rotated_vectors: torch tensor of shape (N, 2)
    """
    base = base_dir / base_dir.norm()
    x_axis = base
    y_axis = torch.tensor([-base[1], base[0]], dtype=torch.float32)
    R = torch.stack([x_axis, y_axis], dim=1)

    return vectors @ R

def polar_feat_from_xy_torch(xy, origin):
    """
    Convert (x,y) to polar features (r, cos(theta), sin(theta)) with respect to origin
    
    Args:
        xy: torch tensor of shape (..., 2)
        origin: torch tensor of shape (2,)
        
    Returns:
        polar_features: torch tensor of shape (..., 3)
    """
    xy = xy.float(); origin = origin.to(xy)
    shifted = xy - origin
    r = torch.sqrt(shifted[..., 0]**2 + shifted[..., 1]**2)
    theta = torch.atan2(shifted[..., 1], shifted[..., 0])

    return torch.stack([r, torch.cos(theta), torch.sin(theta)], dim=-1)

def align_and_scale_gp_prediction(
    ref_traj_np, seed_end, probe_end, K_hist, preds_ref_np, probe_points,
    mode='angle',
    time_scale_override=None,
    time_scale_used_anchors=None,
    spatial_scale_override=None,        
    dtheta_override=None                
):
    """
    Align and scale GP predictions from reference trajectory frame to probe trajectory frame.

    Args:
        ref_traj_np: numpy array of shape (N, 2), reference trajectory points
        seed_end: int, index of the last seed point in reference trajectory
        probe_end: int, index of the last probe point in probe trajectory
        K_hist: int, history length
        preds_ref_np: numpy array of shape (M, 2), GP predicted points in reference frame
        probe_points: list or numpy array of shape (L, 2), probe trajectory points
        mode: str, 'angle' or 'manual'
        time_scale_override: float or None, override time scale if provided
        time_scale_used_anchors: int or None, number of anchors used for time scale estimation
        spatial_scale_override: float or None, override spatial scale if provided
        dtheta_override: float or None, override rotation angle if provided

    Returns:
        preds_new: numpy array of shape (M, 2), aligned and scaled predictions in probe frame
        params: dict, parameters used for transformation
    """
    assert seed_end >= K_hist - 1

    ref = np.asarray(ref_traj_np, dtype=np.float64)
    ref_seed = ref_traj_np[seed_end - (K_hist - 1): seed_end + 1]  # Shape: (K, 2)

    probe = np.asarray(probe_points, dtype=np.float64)
    assert probe.shape[0] >= 2, "目标段太短"
    if probe.shape[0] >= K_HIST:
        probe_seed = probe[-K_HIST:, :]
    else:
        probe_seed = resample_to_k(probe, K_HIST)

    ref_start = ref[0]
    ref_anchor = ref[int(seed_end)]
    new_start = probe[0]
    new_anchor = probe[int(probe_end)]

    # ======================== ANGLE mode ========================
    if mode == 'angle':
        # Ref reference vector (average direction of first 10 segments)
        k_hist_ref = min(10, ref.shape[0]-1)
        dirs_ref = ref[1:k_hist_ref+1] - ref[0]
        v_ref = dirs_ref.mean(axis=0)

        # Probe reference vector (average direction of first 10 segments)
        k_hist_probe = min(10, probe.shape[0]-1)
        dirs_probe = probe[1:k_hist_probe+1] - probe[0]
        v_new = dirs_probe.mean(axis=0)

        ref_vector = ref_anchor - ref_start
        nr = np.linalg.norm(ref_vector)
        new_vector = new_anchor - new_start
        nn = np.linalg.norm(new_vector)
        if nr < 1e-9 or nn < 1e-9:
            raise ValueError("Angle/scale estimation vector too short")
        
        ang_ref = np.arctan2(ref_vector[1], ref_vector[0])
        ang_new = np.arctan2(new_vector[1], new_vector[0])
        dtheta = ((ang_new - ang_ref + np.pi) % (2*np.pi)) - np.pi
        c, s_ = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s_], [s_, c]], dtype=np.float64)

        spatial_scale = float(nn / nr)
        scale = spatial_scale
        if time_scale_override is not None:
            scale = float(time_scale_override)

        t = new_anchor - scale * (R @ ref_anchor)
        preds_new = (scale * (R @ preds_ref_np.T).T + t)

        params = dict(
            mode='angle',
            dtheta=float(dtheta), s=scale, t=t,
            ref_anchor=ref_anchor, new_anchor=new_anchor,
            ref_start=ref_start, new_start=new_start,
            spatial_scale=spatial_scale,
            time_scale=(None if time_scale_override is None else float(time_scale_override)),
            time_scale_used_anchors=(0 if time_scale_used_anchors is None else int(time_scale_used_anchors))
        )

        return preds_new, params

    # ======================== MANUAL mode (manual rotation/scale) ========================
    elif mode == 'manual':
        # Ref reference vector (average direction of first 10 segments)
        k_hist_ref = min(10, ref.shape[0]-1)
        dirs_ref = ref[1:k_hist_ref+1] - ref[0]
        v_ref = dirs_ref.mean(axis=0)

        # Probe reference vector (average direction of first 10 segments)
        k_hist_probe = min(10, probe.shape[0]-1)
        dirs_probe = probe[1:k_hist_probe+1] - probe[0]
        v_new = dirs_probe.mean(axis=0)

        ref_vector = ref_anchor - ref_start
        nr = np.linalg.norm(ref_vector)
        new_vector = new_anchor - new_start
        nn = np.linalg.norm(new_vector)
        if nr < 1e-9 or nn < 1e-9:
            raise ValueError("Angle/scale estimation vector too short")
        
        ang_ref = np.arctan2(ref_vector[1], ref_vector[0])
        ang_new = np.arctan2(new_vector[1], new_vector[0])
        dtheta = ((ang_new - ang_ref + np.pi) % (2*np.pi)) - np.pi
        c, s_ = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s_], [s_, c]], dtype=np.float64)

        spatial_scale = float(nn / nr)
        scale = spatial_scale
        if time_scale_override is not None:
            scale = float(time_scale_override)

        t = new_anchor - scale * (R @ ref_anchor)
        preds_new = (scale * (R @ preds_ref_np.T).T + t)

        params = dict(
            mode='angle',
            dtheta=float(dtheta), s=scale, t=t,
            ref_anchor=ref_anchor, new_anchor=new_anchor,
            ref_start=ref_start, new_start=new_start,
            spatial_scale=spatial_scale,
            time_scale=(None if time_scale_override is None else float(time_scale_override)),
            time_scale_used_anchors=(0 if time_scale_used_anchors is None else int(time_scale_used_anchors))
        )
        
        return preds_new, params

    else:
        raise ValueError("Mode must be 'angle' or 'manual'")
    
def estimate_rigid_transform_2d(
    ref_anchor_pts,
    probe_anchor_pts,
    *,
    mode: str = "least_squares",
    eps: float = 1e-8,
):
    """
    Estimate 2D rotation (Δθ) and isotropic scale that best aligns ref_anchor_pts → probe_anchor_pts in least-squares sense.

    Args:
        ref_anchor_pts : (N, 2) reference anchor points
        probe_anchor_pts : (N, 2) probe anchor points
        mode : currently only 'least_squares'
        eps : numerical stability

    Returns:
        dtheta : float (radians), rotation from ref → probe
        scale : float, isotropic scale
    """
    ref_anchor_pts = np.asarray(ref_anchor_pts, dtype=np.float64)
    probe_anchor_pts = np.asarray(probe_anchor_pts, dtype=np.float64)

    assert ref_anchor_pts.shape == probe_anchor_pts.shape
    assert ref_anchor_pts.ndim == 2 and ref_anchor_pts.shape[1] == 2
    assert ref_anchor_pts.shape[0] >= 2, "At least 2 points required"

    # 1) Remove centroids (translation-free estimation)
    ref_centroid = ref_anchor_pts.mean(axis=0)
    probe_centroid = probe_anchor_pts.mean(axis=0)

    X = ref_anchor_pts - ref_centroid
    Y = probe_anchor_pts - probe_centroid

    # 2) Solve rotation using SVD (Procrustes)
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # Fix possible reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 3) Extract rotation angle
    dtheta = np.arctan2(R[1, 0], R[0, 0])

    # 4) Estimate isotropic scale
    X_rot = X @ R.T
    scale = np.sum(Y * X_rot) / (np.sum(X_rot ** 2) + eps)

    return float(dtheta), float(scale)

def estimate_dtheta_scale_by_multi_angles(
    ref_xy,
    probe_xy,
    *,
    angle_start_deg=5,
    angle_end_deg=30,
    angle_step_deg=5,
    k_hist=10,
    min_r=1e-6,
    weights=None,
):
    """
    Estimate rotation (Δθ) and isotropic scale between ref_xy and probe_xy using multiple angle anchors.

    Args:
        ref_xy : list or numpy array of shape (N, 2), reference trajectory points
        probe_xy : list or numpy array of shape (M, 2), probe trajectory points
        angle_start_deg : float, starting angle in degrees for anchors
        angle_end_deg : float, ending angle in degrees for anchors
        angle_step_deg : float, step size in degrees for anchors
        k_hist : int, history length for angle computation
        min_r : float, minimum radius to consider an anchor valid
        weights : list or numpy array of shape (L,), optional weights for averaging
    
    Returns:
        dtheta_mean : float, estimated rotation (radians) from ref → probe
        scale_mean : float, estimated isotropic scale from ref → probe
        debug : dict, debug information including used angles and individual estimates
    """
    ref_xy = np.asarray(ref_xy, dtype=np.float64)
    probe_xy = np.asarray(probe_xy, dtype=np.float64)

    # 1) Relative angles
    ref_th, ref_mask = angles_relative_to_start_tangent(ref_xy, k_hist=k_hist, min_r=min_r)
    pro_th, pro_mask = angles_relative_to_start_tangent(probe_xy, k_hist=k_hist, min_r=min_r)

    if ref_th is None or pro_th is None:
        raise ValueError("Failed to compute relative angles")

    # 2) Target angles (radians)
    target_angles = np.deg2rad(np.arange(angle_start_deg, angle_end_deg + 1e-9, angle_step_deg))

    dtheta_list = []
    scale_list  = []
    angle_used  = []
    weight_list = []

    ref_origin = ref_xy[0]
    probe_origin = probe_xy[0]

    for i, ang in enumerate(target_angles):
        ref_idx = first_index_reach_threshold(
            ref_th,
            ref_mask,
            ang,
            inclusive=True
        )

        probe_idx = first_index_reach_threshold(
            pro_th,
            pro_mask,
            ang,
            inclusive=True
        )

        if ref_idx is None or probe_idx is None:
            continue

        vr = ref_xy[ref_idx] - ref_origin
        vp = probe_xy[probe_idx] - probe_origin

        nr = np.linalg.norm(vr)
        np_ = np.linalg.norm(vp)

        if nr < min_r or np_ < min_r:
            continue

        th_r = np.arctan2(vr[1], vr[0])
        th_p = np.arctan2(vp[1], vp[0])

        dtheta = (th_p - th_r + np.pi) % (2*np.pi) - np.pi
        scale = np_ / nr

        dtheta_list.append(dtheta)
        scale_list.append(scale)
        angle_used.append(np.rad2deg(ang))

        if weights is not None:
            weight_list.append(weights[i])

    if len(dtheta_list) == 0:
        raise ValueError("No valid angle anchors found")

    dtheta_arr = np.asarray(dtheta_list)
    scale_arr = np.asarray(scale_list)

    # 3) Averaging
    if weights is None:
        dtheta_mean = float(np.mean(dtheta_arr))
        scale_mean = float(np.mean(scale_arr))
    else:
        w = np.asarray(weight_list, dtype=np.float64)
        w = w / (np.sum(w) + 1e-8)
        dtheta_mean = float(np.sum(w * dtheta_arr))
        scale_mean = float(np.sum(w * scale_arr))

    debug = dict(
        angles_deg=angle_used,
        dtheta_list=dtheta_arr,
        scale_list=scale_arr,
    )

    return dtheta_mean, scale_mean, debug
