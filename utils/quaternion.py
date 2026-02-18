import torch
import numpy as np


def quat_between_vectors(a, b):
    """
    Compute the quaternion that rotates vector a to vector b.
    
    Args:
        a: (3,) source vector
        b: (3,) target vector

    Returns:
        q: (4,) quaternion [w, x, y, z] that rotates a
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    dot = np.dot(a, b)

    if dot < -0.999999:
        # Opposite direction
        axis = np.cross(np.array([1,0,0]), a)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(np.array([0,1,0]), a)
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis])

    axis = np.cross(a, b)
    w = np.sqrt((1.0 + dot) * 2.0) / 2.0
    xyz = axis / (2.0 * w)

    q = np.array([w, xyz[0], xyz[1], xyz[2]])
    q /= np.linalg.norm(q)

    return q

def quat_to_rotmat(q):
    """
    Convert quaternion to rotation matrix.

    Args:
        q: (4,) quaternion [w, x, y, z]

    Returns:
        R: (3,3) rotation matrix
    """
    w, x, y, z = q

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])

    return R

def rotmat_to_quat(R):
    """
    Convert rotation matrix to quaternion.

    Args:
        R: (3,3) rotation matrix

    Returns:
        q: (4,) quaternion [w, x, y, z]
    """
    m = R
    t = np.trace(m)

    if t > 0:
        S = np.sqrt(t + 1.0) * 2
        w = 0.25 * S
        x = (m[2,1] - m[1,2]) / S
        y = (m[0,2] - m[2,0]) / S
        z = (m[1,0] - m[0,1]) / S
    else:
        if (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            S = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
            w = (m[2,1] - m[1,2]) / S
            x = 0.25 * S
            y = (m[0,1] + m[1,0]) / S
            z = (m[0,2] + m[2,0]) / S
        elif m[1,1] > m[2,2]:
            S = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
            w = (m[0,2] - m[2,0]) / S
            x = (m[0,1] + m[1,0]) / S
            y = 0.25 * S
            z = (m[1,2] + m[2,1]) / S
        else:
            S = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
            w = (m[1,0] - m[0,1]) / S
            x = (m[0,2] + m[2,0]) / S
            y = (m[1,2] + m[2,1]) / S
            z = 0.25 * S

    q = np.array([w, x, y, z])

    return q / np.linalg.norm(q)

def quat_mul(q1, q2):
    """
    Multiply two quaternions.

    Args:
        q1: (4,) quaternion [w, x, y, z]
        q2: (4,) quaternion [w, x, y, z]

    Returns:
        q: (4,) product quaternion
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return torch.tensor([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=q1.dtype, device=q1.device)

def quat_inv(q):
    """
    Invert a quaternion.

    Args:
        q: (4,) quaternion [w, x, y, z]

    Returns:
        q_inv: (4,) inverse quaternion [w, -x, -y, -z]
    """
    w, x, y, z = q
    return torch.tensor([w, -x, -y, -z], dtype=q.dtype, device=q.device)

def quat_normalize(q):
    """
    Normalize a quaternion to unit length.

    Args:
        q: (4,) quaternion [w, x, y, z]

    Returns:
        q_normalized: (4,) normalized quaternion
    """
    return q / torch.norm(q)

def quat_log(q, eps=1e-9):
    """
    Logarithm map of a quaternion to its tangent space.

    Args:
        q: (4,) quaternion [w, x, y, z]
        eps: numerical stability threshold

    Returns:
        v: (3,) tangent vector representing the rotation
    """
    w, x, y, z = q
    v = torch.tensor([x, y, z], dtype=q.dtype, device=q.device)

    norm_v = torch.norm(v)

    if norm_v < eps:
        return torch.zeros(3, dtype=q.dtype, device=q.device)

    theta = 2 * torch.atan2(norm_v, w)
    axis = v / norm_v

    return axis * theta

def quat_exp(omega, eps=1e-9):
    """
    Exponential map of a tangent vector to a quaternion.
    
    Args:
        omega: (3,) tangent vector representing the rotation
        eps: numerical stability threshold

    Returns:
        q: (4,) quaternion [w, x, y, z]
    """
    theta = np.linalg.norm(omega)

    if theta < eps:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = omega / theta
    half = theta * 0.5

    w = np.cos(half)
    xyz = axis * np.sin(half)

    return np.array([w, xyz[0], xyz[1], xyz[2]])

def quat_slerp(q0, q1, t, eps=1e-9):
    """
    Spherical linear interpolation between two quaternions.

    Args:
        q0: (4,) start quaternion [w, x, y, z]
        q1: (4,) end quaternion [w, x, y, z]
        t: interpolation factor in [0, 1]
        eps: numerical stability threshold

    Returns:
        q: (4,) interpolated quaternion
    """
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.dot(q0, q1)

    # Ensure shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # If very close → linear
    if dot > 1.0 - eps:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    w0 = np.sin((1 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta

    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q)
