import torch
import numpy as np

from geometry.so3 import so3_log
from geometry.transforms import (
    rotate_to_fixed_frame, 
    polar_feat_from_xy_torch, 
    spherical_feat_from_xyz_torch, 
    direction_feat_from_xyz_torch
)


def build_dataset(traj, k, input_type='polar+delta', output_type='delta'):
    """
    Build dataset from trajectory with specified input and output types.
    
    Args:
        traj: torch tensor of shape (T, 2)
        k: history length
        input_type: str, input feature type ('polar', 'delta', 'polar+delta', 'spherical' or 'spherical+delta')
        output_type: str, output type ('delta', 'absolute', 'polar_next')
        
    Returns:
        Xs: torch tensor of shape (N, D_in)
        Ys: torch tensor of shape (N, D_out)
    """
    Xs, Ys = [], []  # Lists to store input-output pairs
    
    T = traj.shape[0]
    global_origin = traj[0]  # Use the first point as global origin
    deltas = traj[1:] - traj[:-1]  # Shape: (T-1, 2)

    # Compute global base direction (average direction of first 10 segments)
    end_idx = min(10, traj.shape[0]-1)  # Prevent insufficient points
    dirs = traj[1:end_idx+1] - traj[0]  # Shape: (end_idx, 2)
    global_base_dir = dirs.mean(dim=0)  # Average direction
    
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

def build_dataset_3d(traj, k, input_type='delta', output_type='delta'):
    """
    Build 3D dataset from trajectory with specified input and output types.

    Initial version:
    - Cartesian only
    - No rotation / polar features
    - Delta-based modeling

    Args:
        traj: torch tensor of shape (T, 3)
        k: history length
        input_type: currently only supports 'delta' or 'pos'
        output_type: 'delta' or 'absolute'

    Returns:
        Xs: torch tensor of shape (N, k*3)
        Ys: torch tensor of shape (N, 3)
    """
    assert traj.ndim == 2 and traj.shape[1] == 3, f"Expected traj shape (T, 3), got {traj.shape}"

    global_origin = traj[0]

    # Precompute deltas
    deltas = traj[1:] - traj[:-1]
    N = deltas.shape[0]

    Xs, Ys = [], []
    for i in range(k, N):
        # Input
        if input_type == 'delta':
            hist = deltas[i-k:i]
            Xs.append(hist.reshape(-1))

        elif input_type == 'pos':
            pos_hist = traj[i-k+1:i+1]
            Xs.append(pos_hist.reshape(-1))

        elif input_type == 'pos+delta':
            pos_hist = traj[i-k+1:i+1]
            delta_hist = deltas[i-k:i]
            Xs.append(torch.cat([pos_hist.reshape(-1), delta_hist.reshape(-1)]))
            
        elif input_type == 'spherical':
            pos_hist = traj[i-k+1:i+1]
            sph = spherical_feat_from_xyz_torch(pos_hist, global_origin)
            Xs.append(sph.reshape(-1))

        elif input_type == 'spherical+delta':
            pos_hist = traj[i-k+1:i+1]
            delta_hist = deltas[i-k:i]
            sph = spherical_feat_from_xyz_torch(pos_hist, global_origin)
            Xs.append(torch.cat([sph.reshape(-1), delta_hist.reshape(-1)]))

        elif input_type == 'dir':
            pos_hist = traj[i-k+1:i+1]
            dir_feat = direction_feat_from_xyz_torch(pos_hist, global_origin).reshape(-1)
            Xs.append(dir_feat)

        else:
            raise ValueError(f"Unsupported input_type for 3D: {input_type}")

        # Output
        if output_type == 'delta':
            Ys.append(deltas[i])
        elif output_type == 'absolute':
            Ys.append(traj[i+1])
        else:
            raise ValueError(f"Unsupported output_type for 3D: {output_type}")

    return torch.stack(Xs), torch.stack(Ys)

def build_dataset_6d(traj_pos: torch.Tensor,
                     traj_rot: torch.Tensor,
                     k: int,
                     input_type: str = 'spherical',
                     output_type: str = 'delta'):
    """
    Build 6D dataset from trajectory with specified input and output types.

    Args:
        traj_pos: torch tensor of shape (T, 3)
        traj_rot: torch tensor of shape (T, 3, 3)
        k: history length
        input_type: str, input feature type ('pos', 'delta', 'pos+delta', 'spherical', 'spherical+delta', 'dir')
        output_type: str, output type ('delta' or 'absolute')

    Returns:
        Xs: torch tensor of shape (N, D_in)
        Ys: torch tensor of shape (N, 6)
    """
    assert traj_pos.shape[0] == traj_rot.shape[0], "[Dataset] Position and rotation trajectories must have the same length!"
    T = traj_pos.shape[0]

    global_origin = traj_pos[0]
    deltas = traj_pos[1:] - traj_pos[:-1]

    Xs, Ys = [], []

    for i in range(k, T - 1):
        # Input features
        if input_type == 'pos':
            pos_hist = traj_pos[i-k+1:i+1]
            Xs.append(pos_hist.reshape(-1))

        elif input_type == 'delta':
            delta_hist = deltas[i-k:i]
            Xs.append(delta_hist.reshape(-1))

        elif input_type == 'pos+delta':
            pos_hist = traj_pos[i-k+1:i+1]
            delta_hist = deltas[i-k:i]
            Xs.append(torch.cat([pos_hist.reshape(-1), delta_hist.reshape(-1)]))

        elif input_type == 'spherical':
            pos_hist = traj_pos[i-k+1:i+1]
            sph = spherical_feat_from_xyz_torch(pos_hist, global_origin)
            Xs.append(sph.reshape(-1))

        elif input_type == 'spherical+delta':
            pos_hist = traj_pos[i-k+1:i+1]
            delta_hist = deltas[i-k:i]
            sph = spherical_feat_from_xyz_torch(pos_hist, global_origin)
            Xs.append(torch.cat([sph.reshape(-1), delta_hist.reshape(-1)]))

        elif input_type == 'dir':
            pos_hist = traj_pos[i-k+1:i+1]
            dir_feat = direction_feat_from_xyz_torch(pos_hist, global_origin).reshape(-1)
            Xs.append(dir_feat)

        else:
            raise ValueError(f"Unsupported input_type for 6D: {input_type}")

        # Output features
        if output_type == 'delta':
            R_t = traj_rot[i]
            R_next = traj_rot[i+1]

            # Angular velocity in body frame
            dR = R_t.T @ R_next
            omega_b = so3_log(dR)

            Ys.append(torch.cat([deltas[i], omega_b], dim=0))

        elif output_type == 'absolute':
            Ys.append(torch.cat([traj_pos[i+1], traj_rot[i+1].reshape(-1)], dim=0))

        else:
            raise ValueError(f"Unsupported output_type for 6D: {output_type}")

    return torch.stack(Xs), torch.stack(Ys)

def build_dataset_cartesian(traj, k):
    """
    Baseline: Input = past k (x,y), Output = next (x,y), all in Cartesian coordinates.

    Args:
        traj: torch tensor of shape (T, 2)
        k: history length

    Returns:
        Xs: torch tensor of shape (N, 2k)
        Ys: torch tensor of shape (N, 2)
    """
    T = traj.shape[0]
    Xs, Ys = [], []
    for t in range(k, T-1):
        seed_pos = traj[t-k+1:t+1]       # Shape: (k, 2)
        Xs.append(seed_pos.reshape(-1))  # Flatten to (2k,)
        Ys.append(traj[t+1])             # Next point (2,)

    return torch.stack(Xs), torch.stack(Ys)

def time_split(X, Y, train_ratio):
    """
    Split dataset into training and testing sets based on time.

    Args:
        X: input data, numpy array of shape (N, D_in)
        Y: output data, numpy array of shape (N, D_out)
        train_ratio: float, ratio of training data
    
    Returns:
        (X_train, Y_train), (X_test, Y_test), ntr
    """
    N = X.shape[0]; ntr = int(N * train_ratio)

    return (X[:ntr], Y[:ntr]), (X[ntr:], Y[ntr:]), ntr
