import torch


def so3_log(R):
    """
    Compute the logarithm map of a rotation matrix R in SO(3).
    
    Args:
        R: torch tensor of shape (3, 3), rotation matrix
    
    Returns:
        omega: torch tensor of shape (3,), rotation vector
    """
    cos_theta = (torch.trace(R) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)

    omega = torch.tensor([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ]) / (2 * torch.sin(theta) + 1e-8)
    
    return theta * omega

def so3_exp(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the exponential map of a rotation vector omega in so(3) using Rodrigues' formula.

    Args:
        omega: torch tensor of shape (3,), rotation vector
        eps: small value to avoid division by zero

    Returns:
        R: torch tensor of shape (3, 3), rotation matrix
    """
    theta = torch.norm(omega)

    if theta < eps:
        return torch.eye(3, dtype=omega.dtype, device=omega.device)

    k = omega / theta
    K = torch.stack([
        torch.stack([torch.tensor(0.0, dtype=omega.dtype, device=omega.device), -k[2], k[1]]),
        torch.stack([k[2], torch.tensor(0.0, dtype=omega.dtype, device=omega.device), -k[0]]),
        torch.stack([-k[1], k[0], torch.tensor(0.0, dtype=omega.dtype, device=omega.device)])
    ])

    R = (
        torch.eye(3, dtype=omega.dtype, device=omega.device)
        + torch.sin(theta) * K
        + (1.0 - torch.cos(theta)) * (K @ K)
    )
    return R
