import torch
import numpy as np

from geometry.frame3d import (
    frame_from_window_transport, 
    local_frame_3d_from_points, 
    rotate_world_to_local_3d
)
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

def build_dataset_3d_local_delta(traj, k, up=(0.0, 0.0, 1.0)):
    """
    Build 3D dataset using local frame + local delta.

    Args:
        traj: torch.Tensor of shape (T, 3)
        k: history length (number of past deltas)
        up: tuple, reference up vector for local frame

    Returns:
        Xs: torch.Tensor of shape (N, k*3)
        Ys: torch.Tensor of shape (N, 3)
    """
    assert traj.ndim == 2 and traj.shape[1] == 3, f"Expected traj shape (T,3), got {traj.shape}"

    traj_np = traj.detach().cpu().numpy()
    T = traj_np.shape[0]

    Xs, Ys = [], []

    # We need at least k+2 points to produce one (X, Y)
    for i in range(k, T - 1):
        # Use points up to i to build frame
        pts_hist = traj_np[i-k:i+1]

        R, _ = frame_from_window_transport(pts_hist, up=up)

        # Build input: past k local deltas
        deltas_local = []
        for j in range(i - k, i):
            dp_world = traj_np[j+1] - traj_np[j]
            dp_local = rotate_world_to_local_3d(dp_world, R)
            deltas_local.append(dp_local)

        X_i = np.concatenate(deltas_local, axis=0)

        # Output: next local delta
        dp_next_world = traj_np[i+1] - traj_np[i]
        y_local = rotate_world_to_local_3d(dp_next_world, R)

        Xs.append(X_i)
        Ys.append(y_local)

    Xs = torch.tensor(np.asarray(Xs), dtype=torch.float32)
    Ys = torch.tensor(np.asarray(Ys), dtype=torch.float32)

    return Xs, Ys

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
