import torch

from geometry.transforms import rotate_to_fixed_frame, polar_feat_from_xy_torch


def build_dataset(traj, k, input_type='polar+delta', output_type='delta'):
    """
    Build dataset from trajectory with specified input and output types.
    
    Args:
        traj: torch tensor of shape (T, 2)
        k: history length
        input_type: str, input feature type ('polar', 'delta', 'polar+delta')
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
