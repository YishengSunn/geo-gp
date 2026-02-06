import numpy as np
import torch

from config.runtime import (
    METHOD_ID, METHOD_HPARAM, 
    MAX_EXPERTS, MAX_DATA_PER_EXPERT, MIN_POINTS_OFFLINE, NEAREST_K, WINDOW_SIZE
)
from geometry.transforms import (
    spherical_feat_from_xyz_torch, 
    direction_feat_from_xyz_torch
)
from gp.skygp_moe import SkyGP_MOE
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

    print(f"Dataset Shape: Xn={Xn.shape}, Yn={Yn.shape}")
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
            dir_feat = direction_feat_from_xyz_torch(hist_pos, global_origin).reshape(1, -1)
            x = dir_feat

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
    gt = traj[start_t + 1 : start_t + 1 + h]

    return preds, gt, h, np.array(vars_seq)
