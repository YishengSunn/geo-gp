import torch

from geometry.transforms import (
    spherical_feat_from_xyz_torch, 
    direction_feat_from_xyz_torch
)


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
