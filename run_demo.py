#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import traceback
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import csv
from skygp_online import SkyGP_MOE as SaoGP_MOE
import matplotlib.cm as cm
import matplotlib.colors as mcolors
# ==============================
# Global Settings
# ==============================
SEED           = 0
SAMPLE_HZ      = 20            # Sample frequency for equal-dt resampling
K_HIST         = 10            # Seed history length
TRAIN_RATIO    = 1.0           # Training ratio (1.0 means full training)

# GP hyperparameters
MAX_EXPERTS    = 40
NEAREST_K      = 2
MAX_DATA_PER_EXPERT = 50
MIN_POINTS_OFFLINE  = 1

# Sliding window size (None means not used)
WINDOW_SIZE    = None
METHOD_ID      = 1             # 1=polar->delta; 5=polar+delta->delta
DOMAIN = dict(xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5)
DEFAULT_SPEED  = 0.2           # Convert polyline length to time for equal-time sampling    
LINE_WIDTHS    = dict(draw=2.0, sampled=1.0, gt=1.0, pred=1.0, seed=1.5, probe=2.0, pred_scaled=1.0)
MATCH_MODE     = 'angle'  # Can switch between similarity / affine / angle (press M key)
MIN_START_ANGLE_DIFF_DEG = 15  # Minimum start angle difference
MIN_START_ANGLE_DIFF = math.radians(MIN_START_ANGLE_DIFF_DEG)
        
# ==== Multi-ref selection related ====
ANCHOR_ANGLE = np.radians(30)  # Anchor angle based on relative start tangent
PHI0_K_PROBE = 500             # Number of initial points used to estimate phi0 (consistent with multi-track version)
PHI0_K_REF   = 500
SELECT_HORIZON = 100           #   Only compare the first few overlapping points' MSE when selecting the best reference

np.random.seed(SEED)
torch.manual_seed(SEED)

# ==============================
# Method Configurations
# ==============================
METHOD_CONFIGS = [
    ('polar',       'delta')
]
METHOD_HPARAM = {
    1: {'adam_lr': 0.001, 'adam_steps': 1}
}

# ==============================
# GP Tools
# ==============================
def torch_to_np(x): return x.detach().cpu().numpy()

class Standardizer:
    def fit(self, X, Y):
        self.X_mean = X.mean(0)
        self.X_std = X.std(0).clamp_min(1e-8)
        self.Y_mean = Y.mean(0)
        self.Y_std = Y.std(0).clamp_min(1e-8)
        return self

    def x_transform(self, X): return (X - self.X_mean) / self.X_std

    def y_transform(self, Y): return (Y - self.Y_mean) / self.Y_std

    def y_inverse_transform(self, Yn):
        assert Yn.shape[-1] == self.Y_std.shape[0], f"维度不匹配: Yn.shape={Yn.shape}, std={self.Y_std.shape}"
        return Yn * self.Y_std + self.Y_mean

    # Compatible with old interface
    def y_inverse(self, Yn): return self.y_inverse_transform(Yn)

# Rotate vectors to a fixed frame defined by base_dir
def rotate_to_fixed_frame(vectors, base_dir):
    base = base_dir / base_dir.norm()
    x_axis = base
    y_axis = torch.tensor([-base[1], base[0]], dtype=torch.float32)
    R = torch.stack([x_axis, y_axis], dim=1)
    return vectors @ R

# Polar features from (x,y) with respect to origin
def polar_feat_from_xy_torch(xy, origin):
    xy = xy.float(); origin = origin.to(xy)
    shifted = xy - origin
    r = torch.sqrt(shifted[...,0]**2 + shifted[...,1]**2)
    theta = torch.atan2(shifted[...,1], shifted[...,0])
    return torch.stack([r, torch.cos(theta), torch.sin(theta)], dim=-1)

# build dataset with desired input/output types
def build_dataset(traj, k, input_type='polar+delta', output_type='delta'):
    deltas = traj[1:] - traj[:-1]
    T = traj.shape[0]
    Xs, Ys = [], []
    global_origin = traj[0]

    ## Compute global base direction
    end_idx = min(10, traj.shape[0]-1)   # Prevent insufficient points
    dirs = traj[1:end_idx+1] - traj[0]   # (end_idx, 2)
    global_base_dir = dirs.mean(dim=0)   # Average direction
    for t in range(k, T-1):
        feats = []
        seed_pos = traj[t-k+1:t+1]
        delta_seq = deltas[t-k+1:t+1]
        if 'polar' in input_type:
            feats.append(polar_feat_from_xy_torch(seed_pos, global_origin).reshape(-1))
        if 'delta' in input_type:
            feats.append(rotate_to_fixed_frame(delta_seq, global_base_dir).reshape(-1))
        Xs.append(torch.cat(feats))
        if output_type == 'delta':
            y_delta = traj[t+1] - traj[t]
            Ys.append(rotate_to_fixed_frame(y_delta.unsqueeze(0), global_base_dir)[0])
        elif output_type == 'absolute':
            Ys.append(traj[t+1].reshape(-1))
        elif output_type == 'polar_next':
            # Predict the next point's polar coordinates relative to the origin
            next_pt = traj[t+1]
            origin = global_origin  # Origin as the polar coordinate origin
            v = next_pt - origin
            r = torch.norm(v)
            theta = torch.atan2(v[1], v[0])
            Ys.append(torch.tensor([r, torch.cos(theta), torch.sin(theta)], dtype=torch.float32))
        else:
            raise ValueError("Unsupported output_type")
    return torch.stack(Xs), torch.stack(Ys)

def build_dataset_cartesian(traj, k):
    """
    Baseline: Input = past k (x,y), Output = next (x,y), all in Cartesian coordinates
    """
    T = traj.shape[0]
    Xs, Ys = [], []
    for t in range(k, T-1):
        seed_pos = traj[t-k+1:t+1]      # (k, 2)
        Xs.append(seed_pos.reshape(-1)) # Flatten to (2k,)
        Ys.append(traj[t+1])            # Next point (2,)
    return torch.stack(Xs), torch.stack(Ys)

def time_split(X, Y, train_ratio):
    N = X.shape[0]; ntr = int(N * train_ratio)
    return (X[:ntr], Y[:ntr]), (X[ntr:], Y[ntr:]), ntr

def train_gp(dataset, method_id=METHOD_ID):
    Xtr = dataset['X_train']; Ytr = dataset['Y_train']
    Din = Xtr.shape[1]
    Dout = Ytr.shape[1]
    scaler = Standardizer().fit(Xtr, Ytr)
    Xn = torch_to_np(scaler.x_transform(Xtr))
    Yn = torch_to_np(scaler.y_transform(Ytr))
    gp_model = SaoGP_MOE(
        x_dim=Din, y_dim=Dout, max_data_per_expert=MAX_DATA_PER_EXPERT,
        nearest_k=NEAREST_K, max_experts=MAX_EXPERTS,
        replacement=False, min_points=10**9, batch_step=10**9,
        window_size=256, light_maxiter=60
    )
    print("Xn.shape:", Xn.shape, "Yn.shape:", Yn.shape)
    for i in range(Xn.shape[0]):
        gp_model.add_point(Xn[i], Yn[i])
    params = METHOD_HPARAM.get(method_id, {'adam_lr':0.001,'adam_steps':200})
    if hasattr(gp_model,"optimize_hyperparams") and params['adam_steps']>0:
        for e in range(len(gp_model.X_list)):
            if gp_model.localCount[e] >= MIN_POINTS_OFFLINE:
                for p in range(2):
                    gp_model.optimize_hyperparams_global(
                        max_iter=params['adam_steps'],
                        verbose=False,
                        window_size=WINDOW_SIZE,
                        adam_lr=params['adam_lr']
                    )
                    
    return {'gp_model': gp_model, 'scaler': scaler, 'input_dim': Din}

def gp_predict(info, feat_1xD):
    gp_model, scaler = info['gp_model'], info['scaler']
    x = torch_to_np(feat_1xD.squeeze(0).float())  # shape: (D,)
    mu, var = gp_model.predict(torch_to_np(scaler.x_transform(torch.tensor(x))))
    mu = np.array(mu).reshape(1, -1)  # ✅ Ensure shape is (1, 2)
    y = torch_to_np(scaler.y_inverse(torch.tensor(mu)))  # shape (1, 2)
    return y, var  # Return numpy with shape (1, 2)

def rollout_reference(model_info, traj, start_t, h, k, input_type, output_type, scaler=None):
    assert start_t >= (k - 1), f"start_t={start_t} 太小，至少需要 {k - 1}"
    T = traj.shape[0]
    h = max(0, h)
    # Keep consistent with training: use global origin and global base_dir
    global_origin = traj[0]
    if traj.shape[0] > 1:
        # print("✅ Compute probe global direction as average direction of first 10 segments")
        end_idx = min(10, traj.shape[0]-1)
        dirs = traj[1:end_idx+1] - traj[0]
        global_base_dir = dirs.mean(dim=0)
    else:
        print("⚠️ Insufficient trajectory points to compute global direction, using default direction")
        global_base_dir = torch.tensor([1.0, 0.0])

    # Initialize historical positions and deltas
    hist_pos = [traj[start_t - (k - 1) + i].clone() for i in range(k)]
    hist_del = []
    for i in range(k):
        idx = start_t - (k - 1) + i
        prev = traj[idx - 1] if idx - 1 >= 0 else traj[0]
        hist_del.append(traj[idx] - prev)

    cur_pos = hist_pos[-1].clone()
    preds_std = []  # Store standardized predictions
    preds_pos = []  # Store actual positions (after inverse standardization)
    vars_seq = []  # New: store variance at each step

    for _ in range(h):
        feats = []

        if 'polar' in input_type:
            # Keep consistent with training: use global_origin
            polar_feat = polar_feat_from_xy_torch(torch.stack(hist_pos[-k:]), global_origin)
            feats.append(polar_feat.reshape(1, -1))  # (1, 2K)

        if 'delta' in input_type:
            # Keep consistent with training: use global_base_dir
            delta_feat = rotate_to_fixed_frame(torch.stack(hist_del[-k:]), global_base_dir)
            feats.append(delta_feat.reshape(1, -1))  # (1, 2(K-1))

        x = torch.cat(feats, dim=1)  # shape (1, D)

        # GP prediction
        y_pred, var = gp_predict(model_info, x)  # Get var
        y_pred = torch.tensor(y_pred, dtype=torch.float32)  # Ensure tensor type consistency
        # print(f"Predicted (std space): {y_pred.numpy()}")
        preds_std.append(y_pred[0])
        vars_seq.append(var)   # Save variance

        # Inverse standardization of output is done only once at the end  
        # During rollout, still use standardized step/delta for calculation
        if output_type == 'delta':
            gb = global_base_dir / global_base_dir.norm()
            R = torch.stack([gb, torch.tensor([-gb[1], gb[0]])], dim=1)
            step_world = y_pred @ R.T  # shape (1, 2)
            next_pos = cur_pos + step_world[0]
            next_del = step_world[0]
        elif output_type == 'polar_next':
            r = y_pred[0, 0]
            cos_t = y_pred[0, 1]
            sin_t = y_pred[0, 2]
            next_pos = global_origin + r * torch.tensor([cos_t, sin_t], dtype=torch.float32)
            next_del = next_pos - cur_pos
        else:
            raise ValueError("Unsupported output_type")
        
        # Update history
        hist_pos.append(next_pos)
        hist_del.append(next_del)
        cur_pos = next_pos
        preds_pos.append(next_pos)
        
    preds = torch.stack(preds_pos, dim=0)

    # Ground truth (optional, for debugging)
    gt = traj[start_t + 1: start_t + 1 + h]
    return preds, gt, h, np.array(vars_seq)

# ==============================
# Sampling & transformation
# ==============================
def resample_polyline_equal_dt(points_xy, sample_hz, speed):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.shape[0] < 2: return pts
    seg = pts[1:]-pts[:-1]; seg_len=np.linalg.norm(seg,axis=1)
    L=float(np.sum(seg_len))
    if L<=1e-8: return pts[:1]
    T_total=L/float(speed); dt=1.0/float(sample_hz)
    t_samples=np.arange(0.0, T_total+1e-9, dt)
    s_samples=(t_samples/T_total)*L
    cum_s=np.concatenate([[0.0], np.cumsum(seg_len)])
    out=[]; j=0
    for s in s_samples:
        while j < len(seg_len)-1 and s > cum_s[j+1]: j+=1
        ds=s-cum_s[j]; r=0.0 if seg_len[j]<1e-9 else ds/seg_len[j]
        p=pts[j]+r*seg[j]; out.append(p)
    return np.asarray(out, dtype=np.float32)

def resample_to_k(points_xy, k):
    pts=np.asarray(points_xy,dtype=np.float64)
    if pts.shape[0] < 2:
        return np.repeat(pts[:1], k, axis=0) if pts.size else np.zeros((k,2),dtype=np.float64)
    seg=pts[1:]-pts[:-1]; seg_len=np.linalg.norm(seg,axis=1)
    cum=np.concatenate([[0.0], np.cumsum(seg_len)]); L=cum[-1]
    if L < 1e-9: return np.tile(pts[:1], (k,1))
    s=np.linspace(0.0, L, k)
    out=[]; j=0
    for si in s:
        while j < len(seg_len)-1 and si > cum[j+1]: j+=1
        ds=si-cum[j]; r=0.0 if seg_len[j]<1e-9 else ds/seg_len[j]
        p=pts[j]+r*seg[j]; out.append(p)
    return np.asarray(out,dtype=np.float64)

# ==============================
# Angle helpers (relative to start tangent)  
# ==============================
def _wrap_pi(a):
    return ((a + np.pi) % (2*np.pi)) - np.pi

def estimate_start_tangent(xy, k: int = 5) -> float:
    """
    Initial heading φ0 by chord-average from the origin:
    mean( P[i] - P[0] ), i=1..k (clamped by available points).
    This matches the 'global_base_dir' used in training/rollout.
    Returns radians in (-pi, pi].
    """
    P = np.asarray(xy, dtype=np.float64)
    if P.shape[0] < 2:
        return 0.0
    # use up to k chords from the origin
    m = int(min(max(1, P.shape[0] - 1), max(1, k)))
    dirs = P[1:m+1] - P[0]
    v = dirs.mean(axis=0)
    if not np.isfinite(v).all() or np.linalg.norm(v) < 1e-12:
        v = P[1] - P[0]
    return float(np.arctan2(v[1], v[0]))

def build_relative_angles(xy, origin_idx=0, min_r=1e-6):
    P = np.asarray(xy, dtype=np.float64)
    N = len(P)
    if N == 0:
        return np.array([], dtype=np.float64)
    sub = P[origin_idx:]
    th_rel_sub, _ = angles_relative_to_start_tangent(sub, k_hist=K_HIST, min_r=min_r)
    out = np.full(N, np.nan, dtype=np.float64)
    out[origin_idx:origin_idx+len(th_rel_sub)] = th_rel_sub
    
    return out

def angle_diff(a, b):
    return _wrap_pi(a - b)

def crossed_multi_in_angle_rel(theta_from, theta_to, anchor_angles):
    """
    Detect whether the angle changes from theta_from to theta_to cross any angles in anchor_angles.
    Supports only 1 or 2 anchor angles.
    Returns: (bool, count)
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

def estimate_similarity_by_vectors_only(anchor_pairs):
    """
    Given several anchor point pairs (pt_ref, pt_probe), estimate the overall rotation angle dtheta and scale.
    Does not require time information t_ref / t_probe, only uses vectors.
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
        return np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])

    dthetas = [angle_between(vr, vp) for vr, vp in zip(v_refs, v_probes)]
    dtheta = np.mean(dthetas)

    # === Calculate scale (average length ratio)
    norms_ref = np.linalg.norm(v_refs, axis=1)
    norms_probe = np.linalg.norm(v_probes, axis=1)
    scales = norms_probe / norms_ref
    scale = np.mean(scales)

    return dtheta, scale, len(v_refs)

# ==============================
# Map reference frame predictions to new trajectory frame
# ==============================
def align_and_scale_gp_prediction(
    ref_traj_np, seed_end, probe_end, K_hist, preds_ref_np, probe_points,
    mode='angle',
    time_scale_override=None,
    time_scale_used_anchors=None,
    spatial_scale_override=None,        
    dtheta_override=None                
):
    assert seed_end >= K_hist - 1
    ref_seed = ref_traj_np[seed_end - (K_hist - 1): seed_end + 1]  # (K,2)
    probe = np.asarray(probe_points, dtype=np.float64)
    if probe.shape[0] >= K_HIST:
        probe_seed = probe[-K_HIST:, :]
    else:
        probe_seed = resample_to_k(probe, K_HIST)

    ref = np.asarray(ref_traj_np, dtype=np.float64)
    assert probe.shape[0] >= 2, "目标段太短"
    ref_start = ref[0]
    ref_anchor = ref[int(seed_end)]
    new_start = probe[0]
    new_anchor = probe[int(probe_end)]

    # ======================== ANGLE mode ========================
    if mode == 'angle':
        # --- ref reference vector (average direction of first 10 segments) ---
        k_hist_ref = min(10, ref.shape[0]-1)
        dirs_ref = ref[1:k_hist_ref+1] - ref[0]
        v_ref = dirs_ref.mean(axis=0)

        # --- probe reference vector (average direction of first 10 segments) ---
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
        # --- ref reference vector (average direction of first 10 segments) ---
        k_hist_ref = min(10, ref.shape[0]-1)
        dirs_ref = ref[1:k_hist_ref+1] - ref[0]
        v_ref = dirs_ref.mean(axis=0)

        # --- probe reference vector (average direction of first 10 segments) ---
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
        raise ValueError("mode must be 'angle' or 'manual'")

def last_window_rel_angles(points, W, min_r=1e-3):
    P = np.asarray(points, dtype=np.float64)
    if P.shape[0] < 2:
        return None, None
    W = int(max(2, min(W if W is not None else 10, P.shape[0])))
    th, m = angles_relative_to_start_tangent(P, k_hist=W, min_r=min_r)
    end = len(th) - 1
    start = max(0, end - (W - 1))
    return th[start:end+1], m[start:end+1]

def angle_diff_mod_pi(a, b):
    """Calculate the minimum difference between two angles, range (-π, π]"""
    return ((a - b + np.pi) % (2 * np.pi)) - np.pi

# ==============================
# Angle change plotting
# ==============================
def plot_angle_changes(ref_pts, probe_pts, k_hist=10, min_r=1e-3):
    """
    ref_pts: (N,2) numpy array, reference trajectory
    probe_pts: (M,2) numpy array, probe trajectory
    k_hist: window length for tangent estimation (consistent with your code's K_HIST)
    """

    # --- Calculate angle sequences ---
    ref_angles, mask_ref = angles_relative_to_start_tangent(ref_pts, k_hist, min_r)
    probe_angles, mask_probe = angles_relative_to_start_tangent(probe_pts, k_hist, min_r)

    # --- Plotting ---
    plt.figure(figsize=(10,4))
    plt.plot(ref_angles, label="Reference traj (relative angle)", color='red')
    plt.plot(probe_angles, label="Probe traj (relative angle)", color='blue')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel("Point index")
    plt.ylabel("Relative angle (rad)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Relative angle changes of Reference vs Probe")
    plt.show()

    return ref_angles, probe_angles

# ==============================
# Mark specific angles and corresponding points and vectors on angle curves and trajectory plots
# ==============================

def angles_relative_to_start_tangent(points, k_hist, min_r=1e-3):
    P = np.asarray(points, dtype=np.float64)
    if len(P) == 0:
        return np.array([]), np.zeros(0, dtype=bool)
    o = P[0]
    phi0 = estimate_start_tangent(P, k=k_hist)
    v = P - o
    r = np.linalg.norm(v, axis=1)
    th = np.arctan2(v[:,1], v[:,0])
    th_rel = _wrap_pi(th - phi0)
    mask = (r > min_r)
    return th_rel, mask

def _angles_with_phi0(points, k_hist, min_r):
    """
    Compatible wrapper:
    - If your angles_relative_to_start_tangent returns (angles, mask, phi0), use it directly;
    - If it only returns (angles, mask), calculate phi0 here.
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

def _compute_base_unit_vec(points, n_segments=10):
    pts = np.asarray(points, dtype=np.float64)
    m = min(n_segments, pts.shape[0]-1)
    if m < 1:
        return np.array([1.0, 0.0], dtype=np.float64)
    seg = np.diff(pts[:m+1], axis=0)                 # first m segments
    n = np.linalg.norm(seg, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    u = seg / n                                      # unit tangent vectors
    v = u.mean(axis=0)                               # average direction
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

    - Angle unit: radians
    - Use minimal signed difference in (-pi, pi] to handle wrapping
    """
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return 0
    if idxs.size == 1:
        return int(idxs[0])

    def _diff(a, b):
        # Minimal signed difference in (-pi, pi]
        return angle_diff_mod_pi(a, b)

    def _bracket_hit(a, b, t) -> bool:
        d  = _diff(b, a)   # minimal rotation from a to b
        dt = _diff(t, a)   # minimal rotation from a to t
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

        if _bracket_hit(a, b, t_pos) or _bracket_hit(a, b, t_neg):
            return int(i_prev)

    # Fallback: choose the one closer to either +target or -target
    diffs_pos = np.array([abs(_diff(float(ang[i]), t_pos)) for i in idxs])
    diffs_neg = np.array([abs(_diff(float(ang[i]), t_neg)) for i in idxs])
    diffs = np.minimum(diffs_pos, diffs_neg)

    k = int(idxs[int(np.argmin(diffs))])
    pos = int(np.where(idxs == k)[0][0])
    return int(idxs[pos - 1]) if pos > 0 else k


# ---------- main ----------
def plot_vectors_at_angle_ref_probe(
    ref_pts, probe_pts, angle_target, *,
    k_hist=10, min_r=1e-3, n_segments_base=10
):
    """
    In two subplots of the same figure, plot:
    - Reference trajectory and probe's target-angle corresponding vector (origin -> matched point)
    - Reference trajectory and probe's "base vector" (average direction of the first n_segments_base segments)

    ref_pts, probe_pts : (N,2)/(M,2)
    angle_target       : target relative angle (rad)
    k_hist             : window for estimating the starting tangent in the angle curve (aligned with your project's K_HIST)
    n_segments_base    : number of initial segments used for the base vector (consistent with your algorithm, default 10)
    """
    ref_pts   = np.asarray(ref_pts,   dtype=np.float64)
    probe_pts = np.asarray(probe_pts, dtype=np.float64)
    assert ref_pts.shape[0] >= 2 and probe_pts.shape[0] >= 2, "ref/probe 点数至少为2"

    # Angle curves + base angles
    ref_ang, ref_mask, _   = _angles_with_phi0(ref_pts,  k_hist=k_hist, min_r=min_r)
    pro_ang, pro_mask, _   = _angles_with_phi0(probe_pts, k_hist=k_hist, min_r=min_r)

    def _masked_nearest_idx(angles, mask, target):
        if not np.any(mask):
            return 0
        idxs = np.where(mask)[0]
        return int(idxs[np.argmin(np.abs(angles[idxs] - target))])

    i_ref = first_index_reach_threshold(ref_ang, ref_mask, angle_target, inclusive=True)
    print("i_ref (by threshold) =", i_ref, "; nearest =", _masked_nearest_idx(ref_ang, ref_mask, angle_target))
    i_pro = first_index_reach_threshold(pro_ang, pro_mask, angle_target, inclusive=True)
    print("i_pro (by threshold) =", i_pro, "; nearest =", _masked_nearest_idx(pro_ang, pro_mask, angle_target))

    # Origin and target vectors
    o_ref, p_ref = ref_pts[0],   ref_pts[i_ref]
    v_ref        = p_ref - o_ref
    o_pro, p_pro = probe_pts[0], probe_pts[i_pro]
    v_pro        = p_pro - o_pro

    # Base unit vectors (strictly average direction of the first 10 segments)
    u_ref = _compute_base_unit_vec(ref_pts,   n_segments=n_segments_base)
    u_pro = _compute_base_unit_vec(probe_pts, n_segments=n_segments_base)

    # Give the base vectors a suitable display length (only affects visualization, does not change direction)
    L_ref = max(np.linalg.norm(v_ref), 1e-6) * 0.6
    L_pro = max(np.linalg.norm(v_pro), 1e-6) * 0.6
    b_ref = u_ref * L_ref
    b_pro = u_pro * L_pro

    # # ---- 绘图 ----
    # fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    # # 左：参考
    # axL.plot(ref_pts[:,0], ref_pts[:,1], '-', label='Reference traj')
    # axL.scatter(o_ref[0], o_ref[1], c='k', s=30, label='Origin')
    # axL.scatter(p_ref[0], p_ref[1], c='g', s=60, marker='x', label=f'target idx={i_ref}')
    # # target 向量
    # axL.plot([o_ref[0], o_ref[0] + v_ref[0]], [o_ref[1], o_ref[1] + v_ref[1]],
    #         linewidth=2, color='g', label='Target vector')
    # # 基准向量（前10段平均方向）
    # axL.plot([o_ref[0], o_ref[0] + b_ref[0]], [o_ref[1], o_ref[1] + b_ref[1]],
    #         linestyle='--', linewidth=2, color='r', label='Base tangent')
    # axL.set_aspect('equal', adjustable='box')
    # axL.grid(True, alpha=0.3)
    # axL.set_title(f"Reference | target={angle_target:.2f} rad ({np.degrees(angle_target):.1f}°)")
    # axL.legend(loc='best', fontsize=9)

    # # 右：probe
    # axR.plot(probe_pts[:,0], probe_pts[:,1], '-', label='Probe traj')
    # axR.scatter(o_pro[0], o_pro[1], c='k', s=30, label='Origin')
    # axR.scatter(p_pro[0], p_pro[1], c='g', s=60, marker='x', label=f'target idx={i_pro}')
    # axR.plot([o_pro[0], o_pro[0] + v_pro[0]], [o_pro[1], o_pro[1] + v_pro[1]],
    #         linewidth=2, color='g', label='Target vector')
    
    # axR.plot([o_pro[0], o_pro[0] + b_pro[0]], [o_pro[1], o_pro[1] + b_pro[1]],
    #         linestyle='--', linewidth=2, color='r', label='Base tangent')
    # axR.set_aspect('equal', adjustable='box')
    # axR.grid(True, alpha=0.3)
    # axR.set_title("Probe (same target definition)")
    # axR.legend(loc='best', fontsize=9)

    # plt.suptitle("Target vector & Base tangent (Reference vs Probe)")
    # plt.tight_layout()
    # plt.show()

    return {
        "ref_index": i_ref,   "ref_vector": v_ref,   "ref_point": p_ref,   "ref_base_unit": u_ref,
        "probe_index": i_pro, "probe_vector": v_pro, "probe_point": p_pro, "probe_base_unit": u_pro
    }

def _closest_index(pt, arr):
    arr = np.asarray(arr, dtype=np.float64)
    d = np.linalg.norm(arr - pt[None, :], axis=1)
    return int(np.argmin(d))

def get_anchor_correspondence(ref_pts, probe_pts, angle_target, *, min_r: float = 1e-3, n_segments_base: int = 10):
    """
    Non-plotting version: Given raw reference/probe trajectories, find the first index where the relative starting tangent angle = ±angle_target is crossed,
    and return the vectors from origin to anchor points and the base unit vectors (average direction of the first n segments) for both trajectories.
    """
    ref_pts = np.asarray(ref_pts, dtype=np.float64)
    probe_pts = np.asarray(probe_pts, dtype=np.float64)
    assert ref_pts.shape[0] >= 2 and probe_pts.shape[0] >= 2, "ref/probe require at least 2 points"

    # Keep consistent with your existing angle functions
    def _angles_with_phi0_local(points, k_hist, min_r):
        out = angles_relative_to_start_tangent(points, k_hist=k_hist, min_r=min_r)
        if isinstance(out, tuple) and len(out) == 3:
            return out
        elif isinstance(out, tuple) and len(out) == 2:
            angles, mask = out
            phi0 = estimate_start_tangent(points, k=k_hist)
            return angles, mask, phi0
        else:
            raise RuntimeError("angles_relative_to_start_tangent returned an unexpected format")

    ref_ang, ref_mask, _ = _angles_with_phi0_local(ref_pts,  k_hist=PHI0_K_REF,   min_r=min_r)
    pro_ang, pro_mask, _ = _angles_with_phi0_local(probe_pts, k_hist=PHI0_K_PROBE, min_r=min_r)

    i_ref = first_index_reach_threshold(ref_ang, ref_mask, angle_target, inclusive=True)
    i_pro = first_index_reach_threshold(pro_ang, pro_mask, angle_target, inclusive=True)

    o_ref, p_ref = ref_pts[0],   ref_pts[i_ref]
    o_pro, p_pro = probe_pts[0], probe_pts[i_pro]
    v_ref = p_ref - o_ref
    v_pro = p_pro - o_pro

    def _base(points, n_segments):
        m = min(n_segments, points.shape[0]-1)
        if m < 1: return np.array([1.0, 0.0], dtype=np.float64)
        seg = np.diff(points[:m+1], axis=0)
        n = np.linalg.norm(seg, axis=1, keepdims=True); n[n<1e-12] = 1.0
        u = seg / n
        v = u.mean(axis=0)
        if np.linalg.norm(v) < 1e-12: v = seg[0]
        return v / max(np.linalg.norm(v), 1e-12)

    u_ref = _base(ref_pts,   n_segments_base)
    u_pro = _base(probe_pts, n_segments_base)

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

# ==============================
# GUI
# ==============================
class DrawGPApp:
    def __init__(self):
        # === Global style ===
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 12,        
            "axes.titlesize": 18,    
            "axes.labelsize": 16, 
            "xtick.labelsize": 14,  
            "ytick.labelsize": 14,
            "axes.edgecolor": "#000000",
            "axes.linewidth": 1.0,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "legend.frameon": True,
            "legend.edgecolor": "#000000",
            "legend.facecolor": "white",
        })

        # === GUI Initialization ===
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6))
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect('equal')
        # self.ax.set_title("Trajectory Prediction (Minimal White Style)")

        # Background pure white
        self.fig.patch.set_facecolor("white")
        self.ax.set_facecolor("white")

        # Grid lines (light gray, thin)
        self.ax.grid(True, color="#DDDDDD", linestyle="--", linewidth=0.6, alpha=0.8)

        # --- State ---
        self.ref_pts=[]
        self.sampled=None
        self.model_info=None
        self.model_info_baseline=None
        self.seed_end=None
        self.probe_end=None
        self.dtheta_manual = 0.0
        self.scale_manual = 1.0
        self.probe_pts=[]
        self.pred_scaled=None
        self.match_mode = MATCH_MODE
        self.refs = []
        self.best_ref = None

        self.anchors = []
        self.probe_predict_mode = 'probe-based'

        self.past_ref_lines = []   # Historical reference Line2D objects
        self.ref_counter = 0       # Counter: Reference #id

        # Prepare a color cycle during class initialization (e.g., 10 colors cycling)
        self.past_colors = ["green"] + ["orange"]
        self.ref_counter = 0
        self.past_ref_lines = []

        self.rollout_horizon = 2000  # baseline rollout horizon

        # ---- Anchors / states ----
        self.anchor_step = 50
        self.anchors = []
        self.anchor_markers = []
        self.show_anchors = False
        self.anchor_count_total = 0
        self.ref_rel_angle = None
        self.last_end_idx = None
        self.current_anchor_ptr = 0
        self.probe_cross_count_session = 0
        self.probe_crossed_set_session = set()
        self.lookahead_buffer = None
        self.anchor_window = K_HIST  # 👈 used in on_move
        self.goal_stop_eps = 0.05    # 👈 used in predict_on_transformed_probe
        
        self.baseline_vars = None  # ✅ variance of baseline rollout
        self.pred_vars = None      # ✅ variance of current prediction
        
        
        # ---- Corresponding vector visualization ----
        self.anchor_vecs_ref = []
        self.anchor_vecs_probe = []

        self.pred_ref = None

        # --- Curves to display ---
        # demo trajectory (reference)
        self.line_ref, = self.ax.plot([], [], '-', color='red', lw=3.0, label='Demonstration')

        # probe trajectory
        self.line_probe, = self.ax.plot([], [], '-', color='black', lw=3.0, label='Prompt')
        self.line_ps, = self.ax.plot([], [], '-', color='blue', lw=3.0, label='Prediction')
        self.line_ref_tmp, = self.ax.plot([], [], '-', color='gray', lw=1.0, visible=False)
        self.line_samp,   = self.ax.plot([], [], '.', color='#FF7F0E', markersize=2, visible=False)
        self.line_seed,   = self.ax.plot([], [], '-', color='black', lw=1.5, visible=False)
        self.line_pred,   = self.ax.plot([], [], '-', color='green', lw=1.0, visible=False)
        self.line_gt,     = self.ax.plot([], [], '-', color='purple', lw=1.0, visible=False)

        self.ax.legend(fontsize=14, loc='upper right')

        # --- Events ---
        self.drawing_left=False
        self.drawing_right=False
        self.cid_press   = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_move    = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.cid_key     = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.tight_layout()
        plt.show(block=True)
        
    def handle_predict_baseline(self):
        if not hasattr(self, "model_info_baseline") or self.model_info_baseline is None:
            print("❗Baseline GP 未训练")
            return
        if self.probe_pts is None or len(self.probe_pts) < K_HIST:
            print("❗probe 太短，至少需要 K_HIST 个点")
            return

        # --- Step 1: probe → ref frame ---
        probe_np = np.asarray(self.probe_pts, dtype=np.float64)

        # --- Step 2: baseline rollout in probe frame ---
        cur_hist = [torch.as_tensor(pt, dtype=torch.float32) for pt in probe_np[-K_HIST:]]
        preds_ref, vars_ref = [], []
        rollout_horizon = 500

        for _ in range(rollout_horizon):
            X = torch.cat(cur_hist[-K_HIST:]).reshape(1, -1)
            y_pred, var_pred = gp_predict(self.model_info_baseline, X)  # mu:(1,2), var:(1,2)
            mu = torch.as_tensor(y_pred[0], dtype=torch.float32)
            preds_ref.append(mu)
            vars_ref.append(var_pred)
            cur_hist.append(mu)

        preds_ref = torch.stack(preds_ref, dim=0).numpy()
        vars_ref = np.array(vars_ref)  # (H,2)

        # --- Step 3: goal = last point of reference trajectory ---
        if self.sampled is not None and preds_ref.shape[0] > 0:
            ref_goal = self.sampled[-1].detach().cpu().numpy()  # reference goal
            dists = np.linalg.norm(preds_ref - ref_goal[None, :], axis=1)
            hits = np.where(dists <= self.goal_stop_eps)[0]
            for cut_idx in hits:
                var_at_hit = np.max(vars_ref[cut_idx])
                print(f"[Baseline] idx={cut_idx} | d={dists[cut_idx]:.3f}, var={var_at_hit:.3f}")
                if var_at_hit > 0.001:   # ✅ Both thresholds met
                    print(f"✂️ Baseline 截断: 命中 ref 终点阈值且方差>0.5 → cut_idx={cut_idx}")
                    preds_ref = preds_ref[:cut_idx]
                    vars_ref = vars_ref[:cut_idx]
                    break

        self.baseline_preds = preds_ref   # ✅ Save baseline rollout trajectory
        self.baseline_vars = vars_ref     # ✅ Save variance of baseline rollout
    
    def predict_on_transformed_probe(self):
        if not hasattr(self, "best_ref") or self.best_ref is None:
            print("❗ Best reference trajectory not found (please draw probe first)")
            return
        if len(self.probe_pts) < K_HIST:
            print("❗ probe 太短")
            return

        # === Step 0: data preparation ===
        ref_np = self.best_ref['sampled'].numpy()
        model_info = self.best_ref['model_info']
        probe_np = np.asarray(self.probe_pts, dtype=np.float64)

        # === Step 1: Δθ and scale ===
        dtheta = self.dtheta_manual
        spatial_scale = self.scale_manual
        print(f"📐 Manually set: Δθ={np.degrees(dtheta):.2f}°, scale={spatial_scale:.3f}")

        # === Step 2: probe → ref frame ===
        c, s = np.cos(-dtheta), np.sin(-dtheta)
        R_inv = np.array([[c, -s], [s, c]])
        probe_origin = probe_np[0]
        probe_in_ref = ((probe_np - probe_origin) @ R_inv.T) / spatial_scale

        # === Step 3: goal (ref last point → probe frame) ===
        c_f, s_f = np.cos(dtheta), np.sin(dtheta)
        R_fwd = np.array([[c_f, -s_f], [s_f, c_f]])
        ref_vec_total = ref_np[-1] - ref_np[0]
        probe_goal = probe_origin + spatial_scale * np.dot(ref_vec_total, R_fwd.T)
        self.probe_goal = probe_goal

        # === Step 4: point-by-point rollout (h=1 per step) ===
        self.pred_scaled = []
        cur_hist = probe_in_ref.copy()  # current history trajectory (in ref frame)
        input_type, output_type = METHOD_CONFIGS[METHOD_ID - 1]
        rollout_horizon = self.rollout_horizon

        for step in range(rollout_horizon):
            preds_ref,_,_, vars_ref = rollout_reference(
                model_info,
                torch.tensor(cur_hist, dtype=torch.float32),
                start_t=cur_hist.shape[0] - 1,
                h=1,  # rollout one step at a time
                k=K_HIST,
                input_type=input_type,
                output_type=output_type
            )

            # Take the last predicted point
            next_ref = preds_ref[-1].numpy()
            cur_hist = np.vstack([cur_hist, next_ref])  # Update history

            # === Transform back to probe frame ===
            next_pos_world = probe_origin + spatial_scale * np.dot(next_ref, R_fwd.T)

            # === Dynamically refresh GUI ===
            self.pred_scaled.append(next_pos_world)
            arr = np.array(self.pred_scaled)
            self.update_scaled_pred(arr)  # Refresh on each append
            plt.pause(0.001)

            # === Truncation logic ===
            if self.probe_goal is not None:
                d = np.linalg.norm(next_pos_world - self.probe_goal)
                if d <= self.goal_stop_eps and np.max(vars_ref) > 0.001:
                    print(f"✂️ Probe-based truncation: step={step}, d={d:.3f}, var={np.max(vars_ref):.3f}")
                    break

        print(f"✅ Probe-based dynamic rollout completed | Δθ={np.degrees(dtheta):.1f}°, scale={spatial_scale:.3f}")

    def _probe_check_cross_current_anchor(self):
        """
        Check if the current probe crosses the anchors of each reference trajectory.
        Each reference trajectory independently maintains its current_anchor_ptr, probe_crossed_set, and lookahead_buffer.
        """
        if len(self.probe_pts) < 2 or not self.refs:
            return 0

        th0 = self.last_probe_angle
        th1, mask = last_window_rel_angles(self.probe_pts, W=self.anchor_window, min_r=1e-3)
        if th1 is None or not mask[-1]:
            return 0

        changed_refs = 0  # Number of trajectories with crossing actions (for counting only)
        cur_probe_idx = len(self.probe_pts) - 1

        for ref in self.refs:
            anchors = ref['anchors']
            ptr = ref['current_anchor_ptr']
            buffer = ref.get('lookahead_buffer', None)

            idx0, idx1, idx2 = ptr, ptr + 1, ptr + 2

            def get_angle(i):
                return anchors[i]['angle'] if i < len(anchors) else None

            crossed0 = crossed_multi_in_angle_rel(th0, th1[-1], [get_angle(idx0)])[0] if idx0 < len(anchors) else False
            crossed1 = crossed_multi_in_angle_rel(th0, th1[-1], [get_angle(idx1)])[0] if idx1 < len(anchors) else False
            crossed2 = crossed_multi_in_angle_rel(th0, th1[-1], [get_angle(idx2)])[0] if idx2 < len(anchors) else False

            # === 1. Cross current anchor idx0 ===
            if crossed0:
                ref['probe_crossed_set'].add(idx0)
                ref['current_anchor_ptr'] = idx0 + 1
                ref['lookahead_buffer'] = None
                anchors[idx0]['probe_idx'] = cur_probe_idx
                changed_refs += 1
                continue

            # === 2. Cross next anchor idx1, set lookahead buffer ===
            elif crossed1:
                print(f"[ref] A{idx1} crossed (lookahead) ⏳")
                ref['lookahead_buffer'] = {
                    'anchor_idx': idx1,
                    'probe_idx': cur_probe_idx
                }

            # === 3. lookahead buffer set, check if crossing idx2 consecutively ===
            if buffer and crossed2:
                k1 = buffer['anchor_idx']
                k2 = idx2

                if 0 <= k1 < len(anchors) and k1 not in ref['probe_crossed_set']:
                    ref['probe_crossed_set'].add(k1)
                    anchors[k1]['probe_idx'] = buffer['probe_idx']
                if 0 <= k2 < len(anchors) and k2 not in ref['probe_crossed_set']:
                    ref['probe_crossed_set'].add(k2)
                    anchors[k2]['probe_idx'] = cur_probe_idx

                ref['current_anchor_ptr'] = k2 + 1
                ref['lookahead_buffer'] = None
                changed_refs += 1

        return changed_refs

    # -------- Anchor visualization --------
    def draw_anchors(self):
        for h in self.anchor_markers:
            try: h.remove()
            except Exception: pass
        self.anchor_markers.clear()

        if not self.show_anchors or self.sampled is None or not self.anchors:
            self.fig.canvas.draw_idle(); return

        ref_np = self.sampled.numpy()
        for k, a in enumerate(self.anchors):
            i = a['idx']
            if 0 <= i < len(ref_np):
                p = ref_np[i]
                m = self.ax.scatter(p[0], p[1], s=20, marker='o', color='black', zorder=4)
                txt = self.ax.text(
                    p[0], p[1],
                    f"A{k}", fontsize=7, color='black',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='black', alpha=0.6),
                    zorder=5
                )
                self.anchor_markers.extend([m, txt])
        self.fig.canvas.draw_idle()

    # -------- Interactive events --------
    def on_press(self, event):
        if event.inaxes != self.ax: return
        if event.button == 1:   # Left button: reference
            self.drawing_left = True
            if not self.line_ref_tmp.get_visible():
                self.line_ref_tmp.set_visible(True)
            self.ref_pts.append([event.xdata, event.ydata])
            self.update_ref_line()
        elif event.button == 3: # Right button: target segment (start new prompt)
            self.drawing_right = True
            # Start a new drawing session: clear probe, reset "within session" counts and sets
            arr = []
            self.update_scaled_pred(arr)  # Refresh on each append
            self.probe_pts = []
            self.update_probe_line()
            self.probe_pts = [[event.xdata, event.ydata]]
            self.update_probe_line()
            self.last_probe_angle = 0.0

            # ✅ New session: reset crossing count and crossed set for this session
            self.probe_cross_count_session = 0
            self.probe_crossed_set_session = set()

            # ✅ Clear old t_probe in anchors (to prevent contamination)
            for a in self.anchors:
                if 't_probe' in a:
                    del a['t_probe']

            # ❌ Do not reset self.current_anchor_ptr (continue from global progress)
            self.current_anchor_ptr = 0  # Do not reset
        
    def on_release(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 1:
            # Left button release: end reference trajectory drawing
            self.drawing_left = False
            return

        if event.button == 3:
            # Right button release: end probe drawing
            self.drawing_right = False

            # ========= 1) Resample probe with equal time intervals (same as reference trajectory) =========
            if len(self.probe_pts) >= 2:
                probe_raw = np.asarray(self.probe_pts, dtype=np.float32)
                # Resample with equal time intervals, same as in handle_train for reference trajectory
                probe_eq = resample_polyline_equal_dt(probe_raw, SAMPLE_HZ, DEFAULT_SPEED)
                # Fallback: keep at least two points
                if probe_eq.shape[0] >= 2:
                    print(f"🪄 probe resampled with equal time intervals: {len(probe_raw)} → {len(probe_eq)} points")
                    self.probe_pts = probe_eq.tolist()
                    self.update_probe_line()
            else:
                print("❗probe has insufficient points for resampling/matching")
                # Continue anyway to let subsequent logic provide clearer errors/prompts

            # ========= 2) Select the reference trajectory that best matches the probe (MSE selection) =========
            best = None
            if hasattr(self, "refs") and self.refs:
                probe_eq_np  = np.asarray(self.probe_pts,     dtype=np.float64)
                probe_raw_np = np.asarray(probe_raw,          dtype=np.float64) if 'probe_raw' in locals() else probe_eq_np
                best_idx, best_pack, best_mse = self._choose_best_ref_by_mse(
                    probe_eq_np, probe_raw_np, horizon=SELECT_HORIZON, align_on_anchor=False
                )
                print(f"🔍 MSE selection: best_idx={best_idx}, best_mse={best_mse:.6f}")
                if best_idx is not None:
                    self.best_ref = self.refs[best_idx]
                    out, dtheta, scale = best_pack
                    self.anchor = out
                    self.dtheta_manual = dtheta
                    self.scale_manual  = scale
                    print(f"🎯Best reference #{best_idx} | MSE@{SELECT_HORIZON}={best_mse:.6f} | "
                        f"Δθ={np.degrees(dtheta):.1f}°, s={scale:.3f}")

                    # Map original anchor points to "resampled indices" for other visualizations
                    ref_resampled   = self.best_ref["sampled"].detach().cpu().numpy()
                    probe_resampled = np.asarray(self.probe_pts, dtype=np.float64)
                    self.seed_end   = _closest_index(self.anchor["ref_point"],   ref_resampled)
                    self.probe_end  = _closest_index(self.anchor["probe_point"], probe_resampled)
                else:
                    self.best_ref = None
                    print("⚠️ MSE selection failed: no available reference")
            else:
                self.best_ref = None
                print("⚠️ No trained reference trajectories available (refs is empty)")

            # ========= 4) Visualization: angle change comparison + reference/target vectors =========
            try:
                if self.sampled is not None and len(self.probe_pts) > 1:
                    ref_np = self.best_ref['sampled'].detach().cpu().numpy()
                    probe_np = np.asarray(self.probe_pts, dtype=np.float64)  # Already resampled probe
                    # Angle change comparison (relative to start tangent)
                    # plot_angle_changes(ref_np, probe_np, k_hist=K_HIST)

                    # Target angle vector visualization (example uses 1 rad, can be connected to UI as needed)
                    angle_target = -0.5
                    out = plot_vectors_at_angle_ref_probe(
                        ref_np, probe_np,
                        angle_target=angle_target,
                        k_hist=K_HIST,
                        n_segments_base=10
                    )
                    self.seed_end = out['ref_index']
                    self.probe_end = out['probe_index']
                    v_ref = out['ref_vector']
                    v_pro = out['probe_vector']

                    self.dtheta_manual = float(np.arctan2(v_pro[1], v_pro[0]) - np.arctan2(v_ref[1], v_ref[0]))
                    self.scale_manual = float(np.linalg.norm(v_pro) / max(np.linalg.norm(v_ref), 1e-6))
                    print(f"📐 Target vector comparison: Δθ={np.degrees(self.dtheta_manual):.1f}°, scale={self.scale_manual:.3f}")
                else:
                    print("❗Insufficient reference or probe points to plot angle/vector comparison")
            except Exception as e:
                print(f"⚠️ Visualization exception: {e}")
                traceback.print_exc()
            
            # ========= 3) Perform matching and scaling prediction (ref-based / probe-based decided internally) =========
            try:
                self.match_and_scale_predict()
            except Exception as e:
                print(f"⚠️ Matching/prediction exception: {e}")
                traceback.print_exc()

            # ========= 5) Clear all temporary probe states in reference trajectories (prepare for next drawing)    =========
            if hasattr(self, "refs"):
                for ref in self.refs:
                    ref['current_anchor_ptr'] = 0
                    ref['probe_crossed_set'] = set()
                    ref['lookahead_buffer'] = None
                    ref['reached_goal'] = False
                    for a in ref.get('anchors', []):
                        a.pop('probe_idx', None)
            print("🧼 Probe states cleared, ready for next drawing")

    def on_move(self, event):
        if event.inaxes != self.ax: return
        if self.drawing_left:
            self.ref_pts.append([event.xdata, event.ydata])
            self.update_ref_line()

        if self.drawing_right:
            self.probe_pts.append([event.xdata, event.ydata])
            self.update_probe_line()
            self._probe_check_cross_current_anchor()
            # === Calculate probe relative angle (relative to probe start tangent) ===
            probe_np = np.asarray(self.probe_pts, dtype=np.float64)
            if probe_np.shape[0] >= 2:
                probe_rel_angle, mask = angles_relative_to_start_tangent(
                    probe_np, k_hist=K_HIST, min_r=1e-6
                )
                if mask[-1]:
                    th_cur = float(probe_rel_angle[-1])
                    self.last_probe_angle = th_cur
        # === Only check if all anchors have been crossed in order, then check if the end angle is crossed ===
        if hasattr(self, 'refs') and self.refs:
            for ref in self.refs:
                # All anchors have been crossed
                if len(ref['probe_crossed_set']) == len(ref['anchors']):
                    final_angle = float(ref['anchors'][-1]['angle'])  # Final anchor angle
                    if not ref.get('reached_goal', False) and len(self.probe_pts) >= 2:
                        th1, mask = angles_relative_to_start_tangent(self.probe_pts, k_hist=K_HIST, min_r=1e-6)
                        if mask[-1]:
                            th_cur = float(th1[-1])
                            crossed, _ = crossed_multi_in_angle_rel(self.last_probe_angle, th_cur, [final_angle])
                            if crossed:
                                ref['reached_goal'] = True
                                print("🎯 All anchors crossed in order, and final angle crossed 🎉! Task completed")

    def on_key(self, event):
        key = event.key.lower()
        if key=='t': self.handle_train()
        elif key=='p': self.handle_predict_reference()
        elif key=='left': self.move_seed(-1)
        elif key=='right': self.move_seed(+1)
        elif key == 'v':
            if self.probe_predict_mode == 'ref-based':
                self.probe_predict_mode = 'probe-based'
            else:
                self.probe_predict_mode = 'ref-based'
            print(f"🔁 Current prediction mode switched to: {self.probe_predict_mode}")
        elif key=='c': self.clear_all()
        elif key=='s': self.save_csv()
        elif key == 'g':  # Directly use probe coordinate system for rollout prediction
            self.predict_on_transformed_probe()
        elif key == 'b':
            self.handle_predict_baseline()
        elif key == 'm':
            self.plot_series_and_mse()
    
        elif key=='n': self.start_new_reference()
        elif key=='a':
            self.show_anchors = not self.show_anchors
            self.draw_anchors()
            print(f"📍 Anchor display: {'ON' if self.show_anchors else 'OFF'} | Total counted={self.anchor_count_total}")

    # -------- Visualization updates --------
    def update_ref_line(self):
        if self.ref_pts:
            pts = np.asarray(self.ref_pts, dtype=np.float32)
            self.line_ref.set_data(pts[:, 0], pts[:, 1])
        else:
            self.line_ref.set_data([], [])
        self.fig.canvas.draw_idle()

    def update_probe_line(self):
        if self.probe_pts:
            pts=np.asarray(self.probe_pts,dtype=np.float32)
            self.line_probe.set_data(pts[:,0], pts[:,1])
        else:
            self.line_probe.set_data([],[])
        self.fig.canvas.draw_idle()

    def update_scaled_pred(self, preds_scaled=None):
        if preds_scaled is not None and len(preds_scaled) > 0:
            arr = np.array(preds_scaled)   # Convert to numpy for plotting
            self.line_ps.set_data(arr[:, 0], arr[:, 1])
            self.pred_scaled = list(map(tuple, preds_scaled))  # Store as list to ensure appendable
        else:
            self.line_ps.set_data([], [])
            self.pred_scaled = []          # Clear to empty list
        self.fig.canvas.draw_idle()


    def update_sample_line(self):
        if self.sampled is not None and len(self.sampled)>0:
            s=self.sampled
            self.line_samp.set_data(s[:,0], s[:,1])
        else:
            self.line_samp.set_data([],[])
        self.fig.canvas.draw_idle()

    def update_seed_line(self):
        if self.sampled is None or self.seed_end is None or self.seed_end < K_HIST-1:
            self.line_seed.set_data([],[])
        else:
            start_idx=self.seed_end-(K_HIST-1)
            seg=self.sampled[start_idx:self.seed_end+1]
            self.line_seed.set_data(seg[:,0], seg[:,1])
        self.fig.canvas.draw_idle()

    def update_ref_pred_gt(self, preds=None, gt=None):
        if preds is not None and len(preds)>0:
            self.line_pred.set_data(preds[:,0], preds[:,1])
        else:
            self.line_pred.set_data([],[])
        if gt is not None and len(gt)>0:
            self.line_gt.set_data(gt[:,0], gt[:,1])
        else:
            self.line_gt.set_data([],[])
        self.fig.canvas.draw_idle()

    def start_new_reference(self):
        """Start a new reference: clear the current reference's temporary state, keep trained refs."""
        self.ref_pts = []
        self.sampled = None
        self.model_info = None
        self.seed_end = None

        self.anchors = []
        self.ref_rel_angle = None
        self.anchor_count_total = 0
        self.current_anchor_ptr = 0
        self.probe_cross_count_session = 0
        self.probe_crossed_set_session = set()

        self.line_ref_tmp.set_data([], [])
        self.line_ref_tmp.set_visible(True)
        self.fig.canvas.draw_idle()
        print("➕ Start a new reference: draw with left mouse button, press T to train when done.")

    # -------- Training/Prediction (Reference Frame) --------
    def handle_train(self):
        if len(self.ref_pts) < 2:
            print("❗Please draw a reference trajectory with the left mouse button (at least 2 points)"); return

        sampled = resample_polyline_equal_dt(self.ref_pts, SAMPLE_HZ, DEFAULT_SPEED)
        if sampled.shape[0] < K_HIST + 2:
            print(f"❗Too few samples {sampled.shape[0]} < {K_HIST+2}"); return
        self.sampled = torch.tensor(sampled, dtype=torch.float32)
        
        input_type, output_type = METHOD_CONFIGS[METHOD_ID-1]
        X, Y = build_dataset(self.sampled, K_HIST, input_type, output_type)
        (Xtr, Ytr), (Xte, Yte), ntr = time_split(X, Y, TRAIN_RATIO)
        ds = {'X_train': Xtr, 'Y_train': Ytr, 'X_test': Xte, 'Y_test': Yte, 'n_train': ntr}
        self.model_info = train_gp(ds, METHOD_ID)

        # self.seed_end = max(K_HIST-1, min(self.sampled.shape[0]-2, int(self.sampled.shape[0]*0.33)))
        self.update_sample_line(); self.update_seed_line(); self.update_ref_pred_gt(None, None)

        # Hide the temporary line drawn while drawing
        self.line_ref_tmp.set_visible(False)

        # If there is an existing current reference line, transfer it to "historical references"
        color_idx = self.ref_counter % len(self.past_colors)
        past_color = self.past_colors[color_idx]
        self.line_ref.set_linewidth(3.0)
        self.line_ref.set_zorder(1)
        self.line_ref.set_color(past_color)
        self.ref_counter += 1
        self.line_ref.set_label(f"Demonstration #{self.ref_counter}")
        self.past_ref_lines.append(self.line_ref)

        # —— Create a new line_ref for the current reference to avoid overwriting past references —— 
        x_new, y_new = [], []
        self.line_ref, = self.ax.plot(
            x_new, y_new, color="red", linewidth=3.0, label="Current Demonstration"
        )
        
        # Do not remove: keep on canvas
        self.ax.legend(fontsize=14, loc='upper right')
        self.fig.canvas.draw_idle()
        
        # —— Build relative angle anchors (every anchor_step points) ——
        ref_np = self.sampled.numpy()
        self.ref_rel_angle = build_relative_angles(ref_np, origin_idx=0, min_r=1e-6)

        self.anchors = []
        step = max(1, int(self.anchor_step))
        anchor_indices = []

        for i in range(step, len(self.ref_rel_angle), step):
            angle = self.ref_rel_angle[i]
            if np.isnan(angle):
                continue
            if len(anchor_indices) == 0:
                # The first anchor point: angle difference with the start point must be greater than the threshold
                angle_diff = abs(angle_diff_mod_pi(angle, 0.0))  # Relative to the start direction
                if angle_diff >= MIN_START_ANGLE_DIFF:
                    anchor_indices.append(i)
            else:
                anchor_indices.append(i)

        # make sure the last point is an anchor
        if (len(self.ref_rel_angle) - 1) not in anchor_indices:
            anchor_indices.append(len(self.ref_rel_angle) - 1)

        # Create anchors
        self.anchors = []
        for i in anchor_indices:
            self.anchors.append({
                'idx': i,
                'angle': float(self.ref_rel_angle[i]),
                't_ref': i / SAMPLE_HZ
            })
        if (len(self.ref_rel_angle)-1) not in [a['idx'] for a in self.anchors]:
            j = len(self.ref_rel_angle)-1
            self.anchors.append({'idx': j, 'angle': float(self.ref_rel_angle[j])})
        # Remove the first point as an anchor
        if self.anchors and self.anchors[0]['idx'] == 0:
            self.anchors = self.anchors[1:]
        self.anchor_count_total = 0
        self.draw_anchors()
        self.last_end_idx = None
        self.current_anchor_ptr = 0
        self.probe_cross_count_session = 0
        self.probe_crossed_set_session = set()
        print(f"📍(relative) Fixed anchors generated: {len(self.anchors)} (step={self.anchor_step})")

        # —— Save the trained reference trajectory to self.refs ——
        self.refs.append(dict(
            sampled=self.sampled,
            model_info=self.model_info,
            anchors=[dict(a) for a in self.anchors],
            current_anchor_ptr=0,
            probe_crossed_set=set(),
            lookahead_buffer=None,
            reached_goal=False,
            raw=np.array(self.ref_pts, dtype=np.float32)    # <<< Added: save the original reference trajectory
        ))
        print(f"🧠 Total trained references: {len(self.refs)}")
        
        # Clear reference trajectory after training
        self.ref_pts.clear()
        self.line_ref.set_data([], [])
        self.fig.canvas.draw_idle()
        print("🚫 All reference trajectories hidden")
    

    def _choose_best_ref_by_mse(
        self,
        probe_eq_np: np.ndarray,
        probe_raw_np: np.ndarray,
        *,
        horizon: int | None = SELECT_HORIZON,
        align_on_anchor: bool = True
    ):
        """
        For each reference in self.refs:
        1) Use the original ref/probe to calculate anchors -> get Δθ and scale
        2) Rotate/scale/translate the "reference evenly sampled trajectory" to the probe coordinate system
        3) Calculate the mean squared error (MSE) of (A - B)^2 by corresponding indices
            - If align_on_anchor=True: align the two sequences using the closest index of the anchor point in the evenly sampled sequence before comparison
            - horizon limits the comparison length
        Returns: (best_idx, (anchor_out, dtheta, scale), best_mse)
        """
        if not hasattr(self, "refs") or len(self.refs) == 0:
            return None, None, float("inf")

        best_idx, best_mse, best_pack = None, float("inf"), None

        for ridx, ref in enumerate(self.refs):
            ref_raw = ref.get("raw", None)
            if ref_raw is None or ref.get("model_info") is None:
                continue

            # 1) Anchors and scale (using original trajectory)
            out = get_anchor_correspondence(
                ref_raw, probe_raw_np, angle_target=ANCHOR_ANGLE, n_segments_base=10
            )
            v_ref, v_pro = out["ref_vector"], out["probe_vector"]

            dtheta = float(np.arctan2(v_pro[1], v_pro[0]) - np.arctan2(v_ref[1], v_ref[0]))
            scale  = float(np.linalg.norm(v_pro) / max(np.linalg.norm(v_ref), 1e-6))

            # 2) Rotate/scale/translate the "reference evenly sampled trajectory" to the probe coordinate system
            ref_samp = ref["sampled"].detach().cpu().numpy()   # (Nr,2)
            c, s = np.cos(dtheta), np.sin(dtheta)
            R = np.array([[c, -s], [s,  c]], dtype=np.float64)
            ref_in_probe = (ref_samp - ref_samp[0]) @ R.T * scale + probe_eq_np[0]

            # 3) Select the aligned overlapping segment and calculate MSE
            if align_on_anchor:
                i_ref_res = _closest_index(out["ref_point"],   ref_samp)
                i_pro_res = _closest_index(out["probe_point"], probe_eq_np)
                offset = int(i_pro_res - i_ref_res)

                start_ref = max(0, -offset)
                start_pro = max(0,  offset)
                n_overlap = min(ref_in_probe.shape[0] - start_ref, probe_eq_np.shape[0] - start_pro)
                if n_overlap <= 0:
                    continue
                if horizon is not None:
                    n_overlap = min(n_overlap, int(horizon))

                A = ref_in_probe[start_ref : start_ref + n_overlap]
                B = probe_eq_np[start_pro : start_pro + n_overlap]
            else:
                n_overlap = min(ref_in_probe.shape[0], probe_eq_np.shape[0])
                if n_overlap <= 0:
                    continue
                if horizon is not None:
                    n_overlap = min(n_overlap, int(horizon))
                A = ref_in_probe[:n_overlap]
                B = probe_eq_np[:n_overlap]

            mse = float(np.mean(np.sum((A - B) ** 2, axis=1)))
            print(f"[MSE] ref#{ridx}: {mse:.6f}")

            if mse < best_mse:
                best_mse  = mse
                best_idx  = ridx
                best_pack = (out, dtheta, scale)

        return best_idx, best_pack, best_mse

    def handle_predict_reference(self):
        if self.model_info is None or self.sampled is None:
            print("❗Please train first (T)"); return
        if self.seed_end is None:
            self.seed_end = K_HIST-1
        input_type, output_type = METHOD_CONFIGS[METHOD_ID-1]
        start_t=int(self.seed_end); h = self.sampled.shape[0] - (start_t+1)
        preds, gt, h_used = rollout_reference(self.model_info, self.sampled, start_t, h, K_HIST, input_type, output_type)
        preds_np = preds.numpy() if preds.numel()>0 else np.zeros((0,2),dtype=np.float32)
        gt_np    = gt.numpy()    if gt.numel()>0    else np.zeros((0,2),dtype=np.float32)
        self.update_ref_pred_gt(preds_np, gt_np)
        mse = float(((preds-gt)**2).mean().item()) if gt.numel()>0 else float('nan')
        print(f"🔮 Reference Prediction: h={h_used} | MSE={mse:.6f}")

    def move_seed(self, delta):
        if self.sampled is None:
            print("❗Please train first (T)"); return
        new_end = (self.seed_end if self.seed_end is not None else (K_HIST-1)) + int(delta)
        self.seed_end = max(K_HIST-1, min(self.sampled.shape[0]-2, new_end))
        self.update_seed_line()
        print(f"↔️ seed_end={self.seed_end}")

    # -------- Matching & Scaling Prediction (including relative angle anchor counting + local seed search) --------
    def match_and_scale_predict(self):
        """
        Two prediction modes:
        - ref-based: rollout on the reference trajectory, then map to the probe coordinate system
        - probe-based: directly use the probe's seed rollout, independent of the reference trajectory
        """

        if self.model_info is None:
            print("❗Please train first (T)")
            return

        input_type, output_type = METHOD_CONFIGS[METHOD_ID - 1]

        if self.probe_predict_mode == 'probe-based':
            self.predict_on_transformed_probe()
            return

        # ========== Mode A: Reference trajectory rollout + mapping ==========
        if self.sampled is None or self.seed_end is None:
            print("❗Missing reference trajectory or seed_end")
            return

        if len(self.probe_pts) < 2:
            print("❗Probe too short")
            return

        try:
            start_t = int(self.seed_end)
            h = self.sampled.shape[0] - (start_t + 1)
            preds_ref, gt_ref, h_used, _ = rollout_reference(
                self.model_info, self.sampled, start_t, h, K_HIST, input_type, output_type
            )
        except Exception as e:
            print(f"⚠️ Reference rollout failed: {e}")
            return

        preds_ref_np = preds_ref.numpy() if preds_ref is not None and preds_ref.numel() > 0 else np.zeros((0,2), dtype=np.float32)
        ref_traj_np = self.sampled.numpy()

        try:
            preds_tar, params = align_and_scale_gp_prediction(
                ref_traj_np=ref_traj_np,
                seed_end=self.seed_end,
                probe_end=self.probe_end,
                K_hist=K_HIST,
                preds_ref_np=preds_ref_np,
                probe_points=self.probe_pts,
                mode=self.match_mode
            )
        except Exception as e:
            print(f"⚠️ Matching failed: {e}")
            return

        self.update_scaled_pred(preds_tar)

        if gt_ref is not None and gt_ref.numel() > 0:
            mse_ref = float(((preds_ref - gt_ref)**2).mean().item())
            pretty = {k: (np.round(v, 4) if isinstance(v, np.ndarray) else v) for k, v in params.items()}
            print(f"🎯 ref-based matching completed | Mode={self.match_mode} | seed_end={self.seed_end} | MSE={mse_ref:.6f} | Params: {pretty}")
        else:
            print("🎯 ref-based matching completed")
            
    def draw_probe_anchors(self):
        # Remove old ones
        for h in self.probe_anchor_markers:
            try:
                h.remove()
            except Exception:
                pass
        self.probe_anchor_markers.clear()

        if len(self.probe_pts) < 1:
            self.fig.canvas.draw_idle()
            return

        pts = np.asarray(self.probe_pts, dtype=np.float64)
        for k, a in enumerate(self.anchors):
            if 't_probe' in a:
                t_probe = a['t_probe']
                idx = int(round(t_probe * SAMPLE_HZ))
                if 0 <= idx < len(pts):
                    p = pts[idx]
                    m = self.ax.scatter(p[0], p[1], s=20, marker='x', color='blue', zorder=5)
                    txt = self.ax.text(
                        p[0], p[1],
                        f"P{k}", fontsize=7, color='blue',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='blue', alpha=0.6),
                        zorder=6
                    )
                    self.probe_anchor_markers.extend([m, txt])

        self.fig.canvas.draw_idle()

    def plot_series_and_mse(self):
        """
        Three plots (all aligned in the probe coordinate system) + export CSV:
        1) X: ref(→probe), probe, pred(probe), baseline
        2) Y: ref(→probe), probe, pred(probe), baseline
        3) Point-wise MSE (starting from idx_start=len(probe)):
        - MSE(pred(probe) vs ref_in_probe)
        - MSE(baseline   vs ref_in_probe)
        Also write all the above data used for plotting into a CSV, with columns aligned based on the global index.
        """
        # --- Basic checks ---
        if self.sampled is None or len(self.sampled) == 0:
            print("❗No reference trajectory sampled (sampled)"); return
        if self.probe_pts is None or len(self.probe_pts) < 1:
            print("❗No probe trajectory"); return
        # --- Data extraction ---
        ref_np = self.sampled.detach().cpu().numpy().astype(np.float64)   # (Nr,2)
        probe_np = np.asarray(self.probe_pts, dtype=np.float64)           # (Np,2)
        pred_probe_world = getattr(self, "pred_scaled", None)             # (Hp,2) prediction (probe-based)
        baseline_world   = getattr(self, "baseline_preds", None)          # (Hb,2) baseline

        pred_probe_world = np.asarray(pred_probe_world, dtype=np.float64) if (pred_probe_world is not None and len(pred_probe_world)>0) else np.zeros((0,2), dtype=np.float64)
        baseline_world   = np.asarray(baseline_world,   dtype=np.float64) if (baseline_world   is not None and len(baseline_world)>0)   else np.zeros((0,2), dtype=np.float64)

        # --- Manual parameters (forward transform from ref to probe) ---
        dtheta = float(getattr(self, "dtheta_manual", 0.0))
        scale  = float(getattr(self, "scale_manual", 1.0))
        c, s = np.cos(dtheta), np.sin(dtheta)
        R_fwd = np.array([[c, -s], [s, c]], dtype=np.float64)

        ref_origin   = ref_np[0]
        probe_origin = probe_np[0]

        # ref → probe：preds_new = scale * (R @ (ref - ref0)) + probe0
        ref_center   = ref_np - ref_origin
        ref_rot      = (R_fwd @ ref_center.T).T
        ref_in_probe = scale * ref_rot + probe_origin   # (Nr,2)

        # --- Index alignment: predictions start from the probe end index ---
        idx_start = len(probe_np)

        # --- Time axes ---
        t_probe      = np.arange(len(probe_np))
        t_ref_probe  = np.arange(len(ref_in_probe))
        t_pred_probe = np.arange(idx_start, idx_start + len(pred_probe_world))
        t_base_probe = np.arange(idx_start, idx_start + len(baseline_world))

        # --- Point-wise MSE (aligned with ref_in_probe, starting from idx_start) ---
        def aligned_mse(pred_xy, ref_xy, start_idx):
            if len(pred_xy) == 0: return np.zeros(0, dtype=np.float64)
            max_len = max(0, len(ref_xy) - start_idx)
            L = int(min(len(pred_xy), max_len))
            if L <= 0: return np.zeros(0, dtype=np.float64)
            diff = pred_xy[:L] - ref_xy[start_idx:start_idx+L]
            return np.mean(diff**2, axis=1)  # (L,)

        mse_probe = aligned_mse(pred_probe_world, ref_in_probe, idx_start)
        mse_base  = aligned_mse(baseline_world,   ref_in_probe, idx_start)

        # --- Plotting ---
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

        # 1) X
        ax = axes[0]
        ax.plot(t_ref_probe, ref_in_probe[:, 0], label="ref→probe_x", linewidth=2)
        ax.plot(t_probe,     probe_np[:, 0],     label="probe_x", linestyle="--")
        if len(pred_probe_world) > 0:
            ax.plot(t_pred_probe, pred_probe_world[:, 0], label="pred(probe)_x", linestyle="-.")
        if len(baseline_world) > 0:
            ax.plot(t_base_probe, baseline_world[:, 0],   label="baseline_x", linestyle=":")
        ax.set_title("X components in PROBE frame (pred starts at probe end index)")
        ax.set_xlabel("Index"); ax.set_ylabel("x")
        ax.grid(True, alpha=0.3); ax.legend(loc="best")

        # 2) Y
        ax = axes[1]
        ax.plot(t_ref_probe, ref_in_probe[:, 1], label="ref→probe_y", linewidth=2)
        ax.plot(t_probe,     probe_np[:, 1],     label="probe_y", linestyle="--")
        if len(pred_probe_world) > 0:
            ax.plot(t_pred_probe, pred_probe_world[:, 1], label="pred(probe)_y", linestyle="-.")
        if len(baseline_world) > 0:
            ax.plot(t_base_probe, baseline_world[:, 1],   label="baseline_y", linestyle=":")
        ax.set_title("Y components in PROBE frame (pred starts at probe end index)")
        ax.set_xlabel("Index"); ax.set_ylabel("y")
        ax.grid(True, alpha=0.3); ax.legend(loc="best")

        # 3) MSE
        ax = axes[2]
        if len(mse_probe) > 0:
            ax.plot(np.arange(idx_start, idx_start + len(mse_probe)), mse_probe,
                    label="MSE(pred(probe) vs ref→probe)")
        if len(mse_base) > 0:
            ax.plot(np.arange(idx_start, idx_start + len(mse_base)),  mse_base,
                    label="MSE(baseline vs ref→probe)")
        ax.set_title("Per-step MSE vs ref→probe (aligned at probe end index)")
        ax.set_xlabel("Index"); ax.set_ylabel("MSE")
        ax.grid(True, alpha=0.3); ax.legend(loc="best")

        plt.show()

        # ---------------- CSV export (aligned by global index) ----------------
        import csv
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"series_mse_{ts}.csv"

        # Construct global index range
        max_index = max(
            len(ref_in_probe)-1,
            len(probe_np)-1,
            (idx_start + len(pred_probe_world) - 1) if len(pred_probe_world) > 0 else -1,
            (idx_start + len(baseline_world)   - 1) if len(baseline_world)   > 0 else -1
        )
        max_index = int(max(0, max_index))

        # Precompute MSE mapping to global index
        mse_probe_map = {}
        for i in range(len(mse_probe)):
            mse_probe_map[idx_start + i] = mse_probe[i]
        mse_base_map = {}
        for i in range(len(mse_base)):
            mse_base_map[idx_start + i] = mse_base[i]

        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            # Header
            w.writerow([
                "index",
                "ref_to_probe_x", "ref_to_probe_y",
                "probe_x", "probe_y",
                "pred_probe_x", "pred_probe_y",
                "baseline_x", "baseline_y",
                "mse_pred_vs_ref_to_probe", "mse_baseline_vs_ref_to_probe"
            ])

            for idx in range(max_index + 1):
                row = [idx]
                # ref→probe
                if idx < len(ref_in_probe):
                    row += [ref_in_probe[idx, 0], ref_in_probe[idx, 1]]
                else:
                    row += ["", ""]
                # probe
                if idx < len(probe_np):
                    row += [probe_np[idx, 0], probe_np[idx, 1]]
                else:
                    row += ["", ""]
                # pred(probe)
                p_i = idx - idx_start
                if 0 <= p_i < len(pred_probe_world):
                    row += [pred_probe_world[p_i, 0], pred_probe_world[p_i, 1]]
                else:
                    row += ["", ""]
                # baseline
                b_i = idx - idx_start
                if 0 <= b_i < len(baseline_world):
                    row += [baseline_world[b_i, 0], baseline_world[b_i, 1]]
                else:
                    row += ["", ""]
                # MSE（逐点）
                row += [
                    mse_probe_map.get(idx, ""),
                    mse_base_map.get(idx, "")
                ]
                w.writerow(row)

        print(f"💾 Exported plotting data and point-wise MSE: {fname}")

    # -------- Miscellaneous --------
    def clear_all(self):
        self.ref_pts.clear(); self.probe_pts.clear()
        self.sampled=None; self.model_info=None; self.seed_end=None; self.probe_end=None

        # —— Clear anchor data and visualization ——
        self.anchors = []
        self.ref_rel_angle = None
        self.anchor_count_total = 0
        for h in getattr(self, "anchor_markers", []):
            try:
                h.remove()
            except Exception:
                pass
        self.anchor_markers.clear()
        self.last_end_idx = None
        self.current_anchor_ptr = 0
        self.probe_cross_count_session = 0
        self.probe_crossed_set_session = set()
        self.probe_prev_contains = False

        
        if getattr(self, "h_goal", None) is not None:
            try: self.h_goal.remove()
            except Exception: pass
            self.h_goal = None
        self.probe_goal = None

        self.line_ref_tmp.set_data([], []); self.line_ref_tmp.set_visible(True)
        if self.line_ref:
            self.line_ref.set_data([], [])
        self.update_probe_line()
        self.update_sample_line()
        self.update_scaled_pred(None)
        self.update_ref_pred_gt(None, None)
        self.update_seed_line()
        # Clear historical reference lines
        if hasattr(self, "past_ref_lines"):
            for ln in self.past_ref_lines:
                try:
                    ln.remove()
                except Exception:
                    pass
            self.past_ref_lines = []
        self.ref_counter = 0
        print("🧹 Cleared all")

    def save_csv(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname_all = f"traj_all_{SAMPLE_HZ}hz_{ts}.csv"
        fname_ref = f"ref_traj_{SAMPLE_HZ}hz_{ts}.csv"   # ✅ Reference trajectory only

        # ---------- Original comprehensive export ----------
        with open(fname_all, "w", newline="") as f:
            w = csv.writer(f)
            # Header
            w.writerow([
                "ref_x", "ref_y",
                "sampled_x", "sampled_y",
                "probe_x", "probe_y",
                "pred_probe_based_x", "pred_probe_based_y",
                "pred_probe_based_varx", "pred_probe_based_vary",
                "baseline_x", "baseline_y",
                "baseline_varx", "baseline_vary"
            ])

            # Various trajectory points (may have different lengths)
            ref = np.asarray(self.ref_pts, dtype=np.float32) if self.ref_pts else np.zeros((0, 2))
            sampled = self.sampled.numpy() if self.sampled is not None else np.zeros((0, 2))
            probe = np.asarray(self.probe_pts, dtype=np.float32) if self.probe_pts else np.zeros((0, 2))

            pred_probe_based = getattr(self, "pred_scaled", None)
            pred_probe_based = pred_probe_based if pred_probe_based is not None else np.zeros((0, 2))
            probe_vars = getattr(self, "pred_vars", None)
            probe_vars = probe_vars if probe_vars is not None else np.zeros((0, 2))

            baseline = getattr(self, "baseline_preds", None)
            baseline = baseline if baseline is not None else np.zeros((0, 2))
            baseline_vars = getattr(self, "baseline_vars", None)
            baseline_vars = baseline_vars if baseline_vars is not None else np.zeros((0, 2))

            # Find the maximum length and write row by row
            max_len = max(len(ref), len(sampled), len(probe), len(pred_probe_based), len(baseline))
            for i in range(max_len):
                row = []
                # Reference trajectory (original hand-drawn)
                row.extend(ref[i].tolist() if i < len(ref) else ["", ""])
                # Evenly timed sampling (for training)
                row.extend(sampled[i].tolist() if i < len(sampled) else ["", ""])
                # probe
                row.extend(probe[i].tolist() if i < len(probe) else ["", ""])
                # probe-based prediction + variance
                if i < len(pred_probe_based):
                    row.extend(pred_probe_based[i].tolist())
                    if i < len(probe_vars):
                        row.extend(probe_vars[i].tolist())
                    else:
                        row.extend(["", ""])
                else:
                    row.extend(["", "", "", ""])
                # baseline prediction + variance
                if i < len(baseline):
                    row.extend(baseline[i].tolist())
                    if i < len(baseline_vars):
                        row.extend(baseline_vars[i].tolist())
                    else:
                        row.extend(["", ""])
                else:
                    row.extend(["", "", "", ""])

                w.writerow(row)

        # ---------- New: Export reference trajectory only ----------
        # Prefer to export the evenly timed sampled reference trajectory (used for training), fallback to original ref_pts if not available
        if self.sampled is not None and len(self.sampled) > 0:
            ref_for_export = self.sampled.detach().cpu().numpy().astype(np.float64)
            header = ["x", "y"]  # Evenly timed sampling
        else:
            ref_for_export = np.asarray(self.ref_pts, dtype=np.float64) if self.ref_pts else np.zeros((0, 2))
            header = ["x", "y"]  # Original hand-drawn

        with open(fname_ref, "w", newline="") as f2:
            w2 = csv.writer(f2)
            w2.writerow(header)
            for i in range(len(ref_for_export)):
                w2.writerow([ref_for_export[i, 0], ref_for_export[i, 1]])

        print(f"💾 Saved comprehensive data: {fname_all}")
        print(f"💾 Saved reference trajectory only: {fname_ref}  (Preferably evenly timed sampling)")


# ==============================
# Entry
# ==============================
if __name__ == "__main__":
    DrawGPApp()