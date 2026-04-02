import numpy as np
import torch

from config.runtime import ( 
    NEAREST_K, MAX_EXPERTS, MAX_DATA_PER_EXPERT, MIN_POINTS_OFFLINE, WINDOW_SIZE,
    METHOD_ID, METHOD_HPARAM
)
from utils.so3 import so3_exp
from geometry.features import (
    spherical_feat_from_xyz_torch, 
    direction_feat_from_xyz_torch
)
from gp.skygp_moe import SkyGP_MOE
from utils.quaternion import rotmat_to_quat, quat_mul, quat_inv, quat_normalize
from utils.misc import torch_to_np, Standardizer


def train_gp(dataset, method_id=METHOD_ID):
    """
    Train GP model on the dataset.

    Args:
        dataset: dict with 'X_train' and 'Y_train' as torch tensors
        method_id: int, method configuration ID
    
    Returns:
        dict with 'gp_model', 'scaler', 'input_dim'
    """
    Xtr = dataset['X_train']; Ytr = dataset['Y_train']
    Din = Xtr.shape[1]
    Dout = Ytr.shape[1]

    scaler = Standardizer().fit(Xtr, Ytr)
    Xn = torch_to_np(scaler.x_transform(Xtr))
    Yn = torch_to_np(scaler.y_transform(Ytr))
    
    gp_model = SkyGP_MOE(
        x_dim=Din, y_dim=Dout, max_data_per_expert=MAX_DATA_PER_EXPERT,
        nearest_k=NEAREST_K, max_experts=MAX_EXPERTS,
        replacement=False, min_points=10**9, batch_step=10**9,
        window_size=256, light_maxiter=60
    )

    print(f"[Train] Dataset Shape: Xn={Xn.shape}, Yn={Yn.shape}")
    for i in range(Xn.shape[0]):
        gp_model.add_point(Xn[i], Yn[i])
    
    params = METHOD_HPARAM.get(method_id, {'adam_lr':0.001, 'adam_steps':200})
    if hasattr(gp_model, "optimize_hyperparams") and params['adam_steps'] > 0:
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
    """
    GP prediction for a single input feature vector.

    Args:
        info: dict with 'gp_model' and 'scaler'
        feat_1xD: torch tensor of shape (1, D_in)
    
    Returns:
        y: numpy array of shape (1, D_out)
        var: variance of prediction
    """
    gp_model, scaler = info['gp_model'], info['scaler']
    x = torch_to_np(feat_1xD.squeeze(0).float())  # Shape: (D_in,)
    
    mu, var = gp_model.predict(torch_to_np(scaler.x_transform(torch.tensor(x))))

    mu = np.array(mu).reshape(1, -1)  # Ensure shape is (1, D_out)
    y = torch_to_np(scaler.y_inverse(torch.tensor(mu)))  # Shape: (1, D_out)

    return y, var  # Return numpy with shape (1, D_out)

def rollout_reference_3d(
    model_info,
    traj,
    start_t,
    h,
    k,
    input_type='delta',
    output_type='delta',
):
    """
    3D rollout trajectory using GP model from a given start time for h steps.

    Args:
        model_info: dict with 'gp_model' and 'scaler'
        traj: torch tensor of shape (T, 3)
        start_t: int, starting time index (position index)
        h: int, rollout horizon (number of future steps)
        k: int, history length (number of past deltas)
        input_type: 'delta', 'pos', 'pos+delta', 'spehrical', or 'spherical+delta'
        output_type: 'delta' or 'absolute'

    Returns:
        preds: torch tensor of shape (h, 3)   # Predicted positions
        gt: torch tensor of shape (h, 3)      # Ground truth positions
        h: int
        vars_seq: numpy array of shape (h,)
    """
    assert traj.ndim == 2 and traj.shape[1] == 3, f"Expected traj shape (T,3), got {traj.shape}"
    assert start_t >= k, f"start_t={start_t} must be >= {k}"

    T = traj.shape[0]
    h = max(0, h)

    deltas = traj[1:] - traj[:-1]  # (T-1, 3)
    global_origin = traj[0]

    preds_pos = []
    vars_seq = []

    # Current position (x_t)
    cur_pos = traj[start_t].clone()

    # Rolling buffers (will be updated autoregressively)
    hist_pos = traj[start_t-k+1:start_t+1].clone()  # (k, 3)
    hist_del = deltas[start_t-k:start_t].clone()  # (k, 3)

    for _ in range(h):
        # Build input x
        if input_type == 'delta':
            x = hist_del.reshape(1, -1)

        elif input_type == 'pos':
            x = hist_pos.reshape(1, -1)

        elif input_type == 'pos+delta':
            x = torch.cat([hist_pos.reshape(-1), hist_del.reshape(-1)], dim=0).reshape(1, -1)

        elif input_type == 'spherical':
            sph = spherical_feat_from_xyz_torch(hist_pos, global_origin)
            x = sph.reshape(1, -1)

        elif input_type == 'spherical+delta':
            sph = spherical_feat_from_xyz_torch(hist_pos, global_origin)
            x = torch.cat([sph.reshape(-1), hist_del.reshape(-1)], dim=0).reshape(1, -1)

        elif input_type == 'dir':
            dir_feat = direction_feat_from_xyz_torch(hist_pos, global_origin)
            x = dir_feat.reshape(1, -1)

        else:
            raise ValueError(f"Unsupported input_type: {input_type}")

        # GP prediction
        y_pred, var = gp_predict(model_info, x)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)[0]  # (3,)
        vars_seq.append(var)

        # Integrate
        if output_type == 'delta':
            delta_next = y_pred
            next_pos = cur_pos + delta_next

        elif output_type == 'absolute':
            next_pos = y_pred
            delta_next = next_pos - cur_pos

        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

        preds_pos.append(next_pos)

        # Update rolling buffers
        hist_pos = torch.cat([hist_pos[1:], next_pos.unsqueeze(0)], dim=0)
        hist_del = torch.cat([hist_del[1:], delta_next.unsqueeze(0)], dim=0)
        cur_pos = next_pos

    preds = torch.stack(preds_pos, dim=0)
    gt = traj[start_t+1:start_t+1+h]

    return preds, gt, h, np.array(vars_seq)

def rollout_reference_6d(
    model_info,
    traj_pos,
    traj_quat,
    start_t,
    h,
    k,
    input_type='spherical',
    output_type='delta',
    R_ref_probe=None,
    traj_force=None,
):
    """
    6D rollout trajectory (position + orientation, and optionally force) using GP model from a given start time for h steps.

    Args:
        model_info: dict with 'gp_model' and 'scaler'
        traj_pos: torch tensor of shape (T, 3), positions
        traj_quat: torch tensor of shape (T, 4), orientations (quaternions)
        start_t: int, starting time index (position index)
        h: int, rollout horizon (number of future steps)
        k: int, history length (number of past deltas)
        input_type: 'delta', 'pos', 'pos+delta', 'spherical', or 'spherical+delta'
        output_type: 'delta' or 'absolute'
        R_ref_probe: (3, 3) rotation matrix, reference to probe frame rotation
        traj_force: optional (T, 3) force aligned with traj_pos; if None, treated as zeros.

    Returns:
        preds_pos: torch tensor of shape (h, 3), predicted positions
        preds_quat: torch tensor of shape (h, 4), predicted orientations (quaternions)
        preds_force: torch tensor of shape (h, 3) if model outputs force, else None
        gt_pos: torch tensor of shape (h, 3), ground truth positions
        gt_quat: torch tensor of shape (h, 4), ground truth orientations (quaternions)
        gt_force: torch tensor of shape (h, 3) if model outputs force, else None
        vars_seq: numpy array of shape (h,), variance at each step
    """
    assert traj_pos.ndim == 2 and traj_pos.shape[1] == 3, f"Expected traj_pos shape (T,3), got {traj_pos.shape}"
    assert traj_quat.ndim == 2 and traj_quat.shape[1] == 4, f"Expected traj_quat shape (T,4), got {traj_quat.shape}"
    assert start_t >= k, f"Expected start_t >= k, got start_t={start_t}, k={k}"

    d_out = int(model_info['scaler'].Y_mean.shape[0])
    predict_force = d_out >= 10

    R_ref_probe = torch.tensor(R_ref_probe, dtype=torch.float32) if R_ref_probe is not None else None

    T = traj_pos.shape[0]
    if traj_force is None:
        traj_force = torch.zeros((T, 3), dtype=traj_pos.dtype, device=traj_pos.device)
    else:
        assert traj_force.shape == (T, 3), f"Expected traj_force (T,3), got {traj_force.shape}"
        traj_force = traj_force.to(device=traj_pos.device, dtype=traj_pos.dtype)

    global_origin = traj_pos[0]

    cur_pos = traj_pos[start_t].clone()
    cur_q = traj_quat[start_t].clone()
    cur_force = traj_force[start_t].clone()

    hist_pos = traj_pos[start_t-k+1:start_t+1].clone()
    hist_del = (traj_pos[1:] - traj_pos[:-1])[start_t-k:start_t].clone()

    preds_pos = []
    preds_quat = []
    preds_force = [] if predict_force else None
    vars_seq = []

    for _ in range(h):
        # Input feature construction
        if input_type == 'delta':
            x_pos = hist_del.reshape(1, -1)

        elif input_type == 'pos':
            x_pos = hist_pos.reshape(1, -1)

        elif input_type == 'pos+delta':
            x_pos = torch.cat([hist_pos.reshape(-1), hist_del.reshape(-1)], dim=0).reshape(1, -1)

        elif input_type == 'spherical':
            sph = spherical_feat_from_xyz_torch(hist_pos, global_origin)
            x_pos = sph.reshape(1, -1)

        elif input_type == 'spherical+delta':
            sph = spherical_feat_from_xyz_torch(hist_pos, global_origin)
            x_pos = torch.cat([sph.reshape(-1), hist_del.reshape(-1)], dim=0).reshape(1, -1)

        elif input_type == 'dir':
            dir_feat = direction_feat_from_xyz_torch(hist_pos, global_origin)
            x_pos = dir_feat.reshape(1, -1)

        else:
            raise ValueError(f"Unsupported input_type: {input_type}")

        # GP prediction (6D output)
        y_pred, var = gp_predict(model_info, x_pos)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)[0]
        vars_seq.append(var)

        if d_out == 7:
            y_pose = y_pred
            d_force_pred = None
        else:
            y_pose = y_pred[:7]
            d_force_pred = y_pred[7:10]

        # Output integration
        if output_type == 'delta':
            delta_world = y_pose[:3]
            next_pos = cur_pos + delta_world

            dq = y_pose[3:7]

            # Ensure shortest representation
            if dq[0] < 0:
                dq = -dq
            dq = quat_normalize(dq)

            # Transform delta to probe frame
            if R_ref_probe is not None:
                q_ref_probe = torch.tensor(rotmat_to_quat(R_ref_probe), dtype=torch.float32)
                dq = quat_mul(quat_mul(quat_inv(q_ref_probe), dq), q_ref_probe)
                dq = quat_normalize(dq)

            # Apply delta (body frame)
            next_q = quat_mul(cur_q, dq)
            next_q = quat_normalize(next_q)

            if predict_force:
                next_force = cur_force + d_force_pred

        elif output_type == 'absolute':
            next_pos = y_pose[:3]
            next_q = quat_normalize(y_pose[3:7])

            if R_ref_probe is not None:
                q_ref_probe = torch.tensor(rotmat_to_quat(R_ref_probe), dtype=torch.float32)
                next_q = quat_mul(q_ref_probe, next_q)
                next_q = quat_normalize(next_q)

            delta_world = next_pos - cur_pos

            if predict_force:
                next_force = d_force_pred

        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

        # Save predictions
        preds_pos.append(next_pos)
        preds_quat.append(next_q)
        if predict_force:
            preds_force.append(next_force)

        # Update history for next step
        hist_pos = torch.cat([hist_pos[1:], next_pos.unsqueeze(0)], dim=0)
        hist_del = torch.cat([hist_del[1:], delta_world.unsqueeze(0)], dim=0)

        cur_pos = next_pos
        cur_q = next_q
        if predict_force and next_force is not None:
            cur_force = next_force

    preds_pos = torch.stack(preds_pos, dim=0)
    preds_quat = torch.stack(preds_quat, dim=0)
    if predict_force:
        preds_force_t = torch.stack(preds_force, dim=0)
    else:
        preds_force_t = None

    gt_pos = traj_pos[start_t+1:start_t+1+h]
    gt_quat = traj_quat[start_t+1:start_t+1+h]
    if predict_force:
        gt_force_t = traj_force[start_t+1:start_t+1+h]
    else:
        gt_force_t = None

    return preds_pos, preds_quat, preds_force_t, gt_pos, gt_quat, gt_force_t, np.array(vars_seq)
