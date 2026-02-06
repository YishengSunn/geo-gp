import torch


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
