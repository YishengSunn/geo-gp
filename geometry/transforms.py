import numpy as np
import torch

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

def spherical_feat_from_xyz_torch(xyz: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    """
    Convert xyz positions to spherical-like features (r, cos(az), sin(az), cos(el), sin(el)) w.r.t. origin.

    Notes:
    - azimuth az = atan2(y, x)
    - elevation el = atan2(z, sqrt(x^2+y^2))
    - Use trig encoding to avoid angle wrap discontinuity.

    Args:
        xyz: torch.Tensor of shape (..., 3)
        origin: torch.Tensor of shape (3,)

    Returns:
        feats: torch.Tensor of shape (..., 5)
    """
    xyz = xyz.float()
    origin = origin.to(xyz)
    v = xyz - origin
    x, y, z = v[..., 0], v[..., 1], v[..., 2]

    r = torch.sqrt(x * x + y * y + z * z)
    rho = torch.sqrt(x * x + y * y)

    az = torch.atan2(y, x)
    el = torch.atan2(z, rho)

    feats = torch.stack([r, torch.cos(az), torch.sin(az), torch.cos(el), torch.sin(el)], dim=-1)
    return feats

def direction_feat_from_xyz_torch(xyz, origin, eps: float = 1e-8):
    """
    Return stable 3D "direction" features [r, ux, uy, uz] for each point.

    This avoids spherical (azimuth/elevation) singularities near the z-axis.

    Args:
        xyz: torch tensor of shape (..., 3)
        origin: torch tensor of shape (3,)
        eps: small value to avoid division by zero

    Returns:
        feat: torch tensor of shape (..., 4) = [r, ux, uy, uz]
    """
    xyz = xyz.float()
    origin = origin.to(xyz)

    v = xyz - origin
    r = torch.norm(v, dim=-1, keepdim=True)  # (..., 1)

    u = v / (r + eps)  # (..., 3)

    return torch.cat([r, u], dim=-1)

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
