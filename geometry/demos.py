import csv
import numpy as np

from utils.quaternion import quat_between_vectors, rotmat_to_quat


def make_ref_force_raw(n: int) -> np.ndarray:
    """
    Build a (N,3) force along global -z (down).

    Args:
        n: int, number of points

    Returns:
        f: (N,3) force along global -z (down)
    """
    n = int(n)
    f = np.zeros((n, 3), dtype=np.float64)
    if n > 0:
        f[:, 2] = np.linspace(0.0, 2.0, n, dtype=np.float64)

    return f

def make_probe_force_raw(n: int) -> np.ndarray:
    """
    Build a (N,3) force along +x, magnitude linear from 0 to 1 along the sequence.

    Args:
        n: int, number of points

    Returns:
        f: (N,3) force along +x, magnitude linear from 0 to 1 along the sequence
    """
    n = int(n)
    f = np.zeros((n, 3), dtype=np.float64)
    if n > 0:
        f[:, 0] = np.linspace(1.0, 2.0, n, dtype=np.float64)

    return f

def load_demo_spirals(app6d, *, T=400, turns=4*np.pi, radius=1.0, speed=0.1):
    """
    Load a demo pair of 3D spirals into ref_raw and probe_raw.

    - Reference: vertical helix around z-axis.
    - Probe: horizontal helix that lies on the drawing plane x=probe_plane_x (roughly),
            but also progresses along +x to mimic "forward motion".

    Args:
        T: int, number of points
        turns: float, total angle in radians for the spirals
        radius: float, radius of the spirals
        speed: float, speed of vertical/horizontal progression
    """
    t = np.linspace(0.0, float(turns), int(T), dtype=np.float64)

    # Reference (vertical helix): (x,y,z) = (cos, sin, speed*t)
    ref = np.stack([radius * np.cos(t), radius * np.sin(t), speed * t], axis=1)

    # Probe (horizontal helix): progress along x, circle in (y,z), keep near x=probe_plane_x
    probe = np.stack(
        [np.full_like(t, app6d.probe_plane_x) + speed * t,
        radius * np.cos(t),
        radius * np.sin(t)],
        axis=1
    )

    app6d.ref_raw = ref.tolist()
    app6d.probe_raw = probe[:50].tolist()

    app6d.update_ref_lines()
    app6d.update_probe_lines()
    app6d.update_pred_lines()

    print(f"[Demo] Loaded spirals: ref_raw={len(app6d.ref_raw)}, probe_raw={len(app6d.probe_raw)}")
    print()

def load_demo_circles_with_orientation(app6d, *, T=400, r_ref=0.5, r_probe=1.0):
    """
    Load a demo pair of circles with orientation into ref_raw/probe_raw and ref_quat_raw/probe_quat_raw.

    - Reference: circle in the xy-plane, centered at origin, with radius r_ref.
    - Probe: circle in the yz-plane, centered at (probe_plane_x, 0, 0), with radius r_probe.

    Args:
        T: int, number of points per circle
        r_ref: float, radius of the reference circle
        r_probe: float, radius of the probe circle
    """
    # Reference
    t = np.linspace(0, 2*np.pi, T)
    ref_pos = np.stack([r_ref*np.cos(t), r_ref*np.sin(t), np.zeros_like(t)], axis=1)

    ref_quat = []
    for p in ref_pos:
        target_dir = -p / np.linalg.norm(p)
        q = quat_between_vectors(np.array([1,0,0]), target_dir)
        ref_quat.append(q)

    ref_quat = np.asarray(ref_quat)

    # Probe
    tp = np.linspace(0, np.pi/2, T//4)
    probe_pos = np.stack([np.zeros_like(tp)+1.5, r_probe*np.cos(tp), r_probe*np.sin(tp)], axis=1)

    probe_quat = []
    center = np.array([1.5, 0.0, 0.0])
    plane_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    for p in probe_pos:
        x_axis = center - p
        x_axis /= np.linalg.norm(x_axis)

        y_axis = plane_normal.copy()

        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        q = rotmat_to_quat(R)
        probe_quat.append(q)

    probe_quat = np.asarray(probe_quat, dtype=np.float64)

    app6d.ref_raw = ref_pos.tolist()
    app6d.probe_raw = probe_pos.tolist()

    app6d.ref_quat_raw = ref_quat
    app6d.probe_quat_raw = probe_quat

    app6d.update_ref_lines()
    app6d.update_probe_lines()
    app6d.update_pred_lines()

    print(f"[Demo] Loaded circles with quaternion orientation: ref_raw={len(app6d.ref_raw)}, probe_raw={len(app6d.probe_raw)}")
    print()

def load_ref_from_csv(app6d, filepath):
    """
    Load position and quaternion from CSV with fields:
    time,x,y,z,qx,qy,qz,qw

    Quaternion will be converted to [w, x, y, z].

    Args:
        filepath: str, path to the CSV file
    """
    pts = []
    quats = []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"])

                qx = float(row["qx"])
                qy = float(row["qy"])
                qz = float(row["qz"])
                qw = float(row["qw"])

                pts.append([x, y, z])
                quats.append([qw, qx, qy, qz])

            except Exception:
                continue

    if len(pts) == 0:
        print("[LoadCSV] No valid rows found!")
        print()
        return

    app6d.ref_raw = pts
    app6d.ref_quat_raw = np.asarray(quats, dtype=np.float64)

    app6d.update_ref_lines()

    print(f"[LoadCSV] Loaded {len(app6d.ref_raw)} reference points with orientation.")
    print()

def load_probe_from_csv(app6d, filepath):
    """
    Load position and quaternion from CSV with fields:
    time,x,y,z,qx,qy,qz,qw

    Quaternion will be converted to [w, x, y, z].

    Args:
        filepath: str, path to the CSV file
    """
    pts = []
    quats = []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"])

                qx = float(row["qx"])
                qy = float(row["qy"])
                qz = float(row["qz"])
                qw = float(row["qw"])

                pts.append([x, y, z])
                quats.append([qw, qx, qy, qz])

            except Exception:
                continue

    if len(pts) == 0:
        print("[LoadCSV] No valid rows found!")
        print()
        return

    app6d.probe_raw = pts
    app6d.probe_quat_raw = np.asarray(quats, dtype=np.float64)

    app6d.update_probe_lines()

    print(f"[LoadCSV] Loaded {len(app6d.probe_raw)} probe points with orientation.")
    print()
