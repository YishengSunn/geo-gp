"""Plot a 3D trajectory schematic with windowed input and frame annotations."""

import numpy as np
import matplotlib.pyplot as plt


W = 20  # Number of past points ending at p(t).
NEXT_POINT_OFFSET = 26  # Visual spacing used for p(t+1) in the schematic.


def generate_trajectory(num_points: int = 400) -> np.ndarray:
    """Generate a smooth synthetic 3D trajectory.

    Args:
        num_points: Number of trajectory samples to generate.

    Returns:
        Array of shape ``(num_points, 3)`` containing xyz positions.
    """
    t = np.linspace(0.0, 1.0, num_points)

    x = 1.82 * t + 0.20 * np.sin(2.8 * np.pi * t)
    y = 0.95 * np.sin(1.26 * np.pi * t) * (0.36 + 0.82 * t)
    z = 1.06 * np.sin(1.08 * np.pi * t) + 0.42 * t + 0.12 * np.sin(2.2 * np.pi * t)

    x -= x[0]
    y -= y[0]
    z -= z[0]

    bend_idx = min(70, num_points - 1)
    bend_end = bend_idx / (num_points - 1)
    bend_phase = np.clip(t / bend_end, 0.0, 1.0)
    bend_weight = np.sin(np.pi * bend_phase) ** 2
    bend_weight[t > bend_end] = 0.0

    z += 0.26 * bend_weight

    return np.column_stack((x, y, z))


def generate_quaternions(pos: np.ndarray) -> np.ndarray:
    """Generate body orientations with x-axis aligned to the path tangent.

    Args:
        pos: Array of shape ``(N, 3)`` containing trajectory positions.

    Returns:
        Array of shape ``(N, 4)`` containing quaternions in wxyz order.
    """
    tangents = np.gradient(pos, axis=0)
    up = np.array([0.0, 0.0, 1.0])
    quats = []
    for t in tangents:
        x = t / (np.linalg.norm(t) + 1e-8)
        y = np.cross(up, x)
        if np.linalg.norm(y) < 1e-6:
            y = np.cross(np.array([0.0, 1.0, 0.0]), x)
        y /= np.linalg.norm(y) + 1e-8
        z = np.cross(x, y)
        R = np.column_stack([x, y, z])
        quats.append(_rotmat_to_quat(R))
    return np.asarray(quats, dtype=np.float64)


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a normalized quaternion.

    Args:
        R: Rotation matrix of shape ``(3, 3)``.

    Returns:
        Quaternion in wxyz order.
    """
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / (np.linalg.norm(q) + 1e-12)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion to a rotation matrix.

    Args:
        q: Quaternion in wxyz order.

    Returns:
        Rotation matrix of shape ``(3, 3)``.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def set_equal_axes(ax, points: np.ndarray, padding: float = 0.03) -> float:
    """Set equal 3D axis ranges around the provided points.

    Args:
        ax: Matplotlib 3D axis to update.
        points: Array of shape ``(N, 3)`` used to determine axis bounds.
        padding: Fractional padding added around the bounding radius.

    Returns:
        Axis radius after padding.
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius *= 1.0 + padding

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    return radius


def draw_body_frame(ax, pos: np.ndarray, quat: np.ndarray, *, scale: float = 0.14, alpha: float = 1.0) -> None:
    """Draw a local body frame at a trajectory point.

    Args:
        ax: Matplotlib 3D axis to draw on.
        pos: Position of the frame origin.
        quat: Quaternion in wxyz order.
        scale: Length of each frame axis.
        alpha: Transparency applied to arrows and labels.
    """
    R = _quat_to_rotmat(quat)
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    labels = ("x'", "y'", "z'")
    for j, (c, lab) in enumerate(zip(colors, labels)):
        d = R[:, j] * scale
        ax.quiver(
            pos[0], pos[1], pos[2],
            d[0], d[1], d[2],
            color=c, alpha=alpha, linewidth=2.0, arrow_length_ratio=0.22,
        )
        tip = pos + d * 1.12
        ax.text(tip[0], tip[1], tip[2], lab, color=c, fontsize=9, alpha=alpha)


def draw_spherical_example(ax, point: np.ndarray) -> None:
    """Draw spherical-coordinate guides for a point.

    Args:
        ax: Matplotlib 3D axis to draw on.
        point: Position vector whose spherical components are visualized.
    """
    px, py, pz = point
    r = np.linalg.norm(point)
    rho = np.hypot(px, py)
    phi = np.arctan2(py, px)
    psi = np.arctan2(pz, rho)

    ax.plot([0.0, px], [0.0, py], [0.0, pz], "--", color="tab:red", linewidth=1.4)
    ax.plot([0.0, px], [0.0, py], [0.0, 0.0], "--", color="tab:green", linewidth=1.1)
    ax.plot([px, px], [py, py], [0.0, pz], ":", color="0.45", linewidth=1.0)

    phi_radius = max(0.42 * rho, 0.20)
    phi_values = np.linspace(0.0, phi, 80)
    phi_arc = np.column_stack(
        (
            phi_radius * np.cos(phi_values),
            phi_radius * np.sin(phi_values),
            np.zeros_like(phi_values),
        )
    )
    ax.plot(phi_arc[:, 0], phi_arc[:, 1], phi_arc[:, 2], color="tab:purple", linewidth=1.4)

    psi_radius = max(0.22 * r, 0.10)
    psi_values = np.linspace(0.0, psi, 80)
    psi_arc = np.column_stack(
        (
            psi_radius * np.cos(psi_values) * np.cos(phi),
            psi_radius * np.cos(psi_values) * np.sin(phi),
            psi_radius * np.sin(psi_values),
        )
    )
    ax.plot(psi_arc[:, 0], psi_arc[:, 1], psi_arc[:, 2], color="tab:brown", linewidth=1.4)

    ax.text(
        0.50 * px - 0.06, 0.50 * py + 0.06, 0.50 * pz + 0.05,
        "r", color="tab:red", fontsize=14,
    )
    ax.text(
        phi_arc[-1, 0] + 0.20,
        phi_arc[-1, 1] - 0.05,
        phi_arc[-1, 2] - 0.01,
        r"$\varphi$", color="tab:purple", fontsize=14, fontweight="bold",
    )
    ax.text(
        psi_arc[-1, 0] + 0.20,
        psi_arc[-1, 1] - 0.05,
        psi_arc[-1, 2] - 0.02,
        r"$\psi$", color="tab:brown", fontsize=14, fontweight="bold",
    )


def draw_origin_axes(ax, axis_length: float = 1.36) -> None:
    """Draw xyz axes at the global origin.

    Args:
        ax: Matplotlib 3D axis to draw on.
        axis_length: Length of each origin axis arrow.
    """
    axes = (
        ((axis_length, 0.0, 0.0), "#d62728", "x"),
        ((0.0, axis_length, 0.0), "#2ca02c", "y"),
        ((0.0, 0.0, axis_length), "#1f77b4", "z"),
    )
    for direction, color, label in axes:
        ax.quiver(
            0.0, 0.0, 0.0,
            direction[0], direction[1], direction[2],
            color=color, linewidth=1.6, arrow_length_ratio=0.14,
        )
        label_scale = 1.06 if label == "x" else 1.03
        label_pos = np.array(direction) * label_scale
        ax.text(label_pos[0], label_pos[1], label_pos[2], label, color=color, fontsize=10)


def main() -> None:
    """Create and display the 3D trajectory schematic."""
    trajectory = generate_trajectory()
    quaternions = generate_quaternions(trajectory)

    t_idx = 70
    w = W
    assert t_idx >= w - 1 and t_idx + NEXT_POINT_OFFSET < len(trajectory)

    p_t = trajectory[t_idx]
    p_next_idx = t_idx + NEXT_POINT_OFFSET
    p_next = trajectory[p_next_idx]
    window_start_idx = t_idx - w + 1
    window_seg = trajectory[window_start_idx : t_idx + 1]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Draw the reference trajectory around the highlighted input window.
    ax.plot(
        trajectory[: window_start_idx + 1, 0],
        trajectory[: window_start_idx + 1, 1],
        trajectory[: window_start_idx + 1, 2],
        color="royalblue", linewidth=2.2, label="Reference trajectory",
    )
    ax.plot(
        trajectory[t_idx:, 0], trajectory[t_idx:, 1], trajectory[t_idx:, 2],
        color="royalblue", linewidth=2.2,
    )

    # Highlight the input window ending at p(t).
    ax.plot(
        window_seg[:, 0], window_seg[:, 1], window_seg[:, 2],
        color="darkorange", linewidth=1.0, label=rf"Input window ($w={w}$)",
    )
    ax.scatter(
        window_seg[:-1, 0], window_seg[:-1, 1], window_seg[:-1, 2],
        color="orange", s=6, alpha=0.65, zorder=4,
    )

    # Mark the start point, current point, and next target point.
    ax.scatter(*trajectory[0], color="black", s=55, label="Start")
    ax.scatter(*p_t, color="darkorange", s=65, edgecolors="k", linewidths=0.5, zorder=6)
    ax.scatter(*p_next, color="crimson", s=65, edgecolors="k", linewidths=0.5, zorder=6)

    ax.text(p_t[0] + 0.06, p_t[1] + 0.03, p_t[2] + 0.03, r"$p(t)$", color="darkorange", fontsize=11, fontweight="bold")
    ax.text(p_next[0] + 0.045, p_next[1] + 0.03, p_next[2] + 0.03, r"$p(t\!+\!1)$", color="crimson", fontsize=11, fontweight="bold")

    # Draw body orientations at p(t) and p(t+1).
    draw_body_frame(ax, p_t, quaternions[t_idx], scale=0.13, alpha=1.0)
    draw_body_frame(ax, p_next, quaternions[p_next_idx], scale=0.13, alpha=0.75)

    # Draw spherical features for p(t) relative to the origin.
    draw_spherical_example(ax, p_t)

    draw_origin_axes(ax)
    set_equal_axes(ax, np.vstack((trajectory, np.zeros((1, 3)))))

    ax.set_title("3D Trajectory: windowed history input and incremental output")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.view_init(elev=26, azim=-48)

    plt.tight_layout()
    # Uncomment to save the figure instead of displaying it only.
    # plt.savefig("3d_trajectory.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
