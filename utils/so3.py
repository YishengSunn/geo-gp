import torch
import numpy as np


def is_torch(x):
    """
    Check if x is a torch.Tensor.
    
    Args:
        x: input object
    
    Returns:
        bool: True if x is a torch.Tensor, False otherwise"""
    return isinstance(x, torch.Tensor)

def eye3_like(x):
    """
    Create a 3x3 identity matrix with the same type (torch or numpy) as x.

    Args:
        x: input object (torch.Tensor or np.ndarray)
    
    Returns:
        3x3 identity matrix of the same type as x
    """
    if is_torch(x):
        return torch.eye(3, dtype=x.dtype, device=x.device)
    else:
        return np.eye(3, dtype=x.dtype)

def so3_log(R, eps: float = 1e-8):
    """
    Log map from SO(3) -> so(3).
    Supports both torch.Tensor and np.ndarray.
    Output type matches input type.

    Args:
        R: (3,3) rotation matrix (torch.Tensor or np.ndarray)
        eps: float, numerical stability threshold

    Returns:
        omega: (3,) rotation vector in so(3) (torch.Tensor or np.ndarray
    """
    use_torch = is_torch(R)

    if use_torch:
        tr = torch.trace(R)
        cos_theta = (tr - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_theta)

        omega_hat = torch.stack([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])

        denom = 2.0 * torch.sin(theta) + eps
        omega = omega_hat / denom
        return theta * omega

    else:
        tr = np.trace(R)
        cos_theta = (tr - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
        theta = np.arccos(cos_theta)

        omega_hat = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ], dtype=R.dtype)

        denom = 2.0 * np.sin(theta) + eps
        omega = omega_hat / denom
        return theta * omega

def so3_exp(omega, eps: float = 1e-8):
    """
    Exponential map from so(3) -> SO(3).
    Supports both torch.Tensor and np.ndarray.
    Output type matches input type.

    Args:
        omega: (3,) rotation vector in so(3) (torch.Tensor or np.ndarray)
        eps: float, numerical stability threshold

    Returns:
        R: (3,3) rotation matrix (torch.Tensor or np.ndarray)
    """
    use_torch = is_torch(omega)

    if use_torch:
        theta = torch.norm(omega)
        if theta < eps:
            return eye3_like(omega)

        k = omega / theta
        K = torch.stack([
            torch.stack([torch.tensor(0.0, dtype=omega.dtype, device=omega.device), -k[2], k[1]]),
            torch.stack([k[2], torch.tensor(0.0, dtype=omega.dtype, device=omega.device), -k[0]]),
            torch.stack([-k[1], k[0], torch.tensor(0.0, dtype=omega.dtype, device=omega.device)]),
        ])

        I = eye3_like(omega)
        R = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)
        return R

    else:
        theta = np.linalg.norm(omega)
        if theta < eps:
            return eye3_like(omega)

        k = omega / theta
        K = np.array([
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ], dtype=omega.dtype)

        I = eye3_like(omega)
        R = I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
        return R
