import csv
import numpy as np

from utils.quaternion import quat_between_vectors, rotmat_to_quat


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

    # Probe (horizontal-ish helix): progress along x, circle in (y,z), keep near x=probe_plane_x
    probe = np.stack(
        [np.full_like(t, app6d.probe_plane_x) + speed * t,
        radius * np.cos(t),
        radius * np.sin(t)],
        axis=1
    )

    app6d.ref_raw = ref.tolist()
    app6d.probe_raw = probe[:50].tolist()

    # Reset derived buffers / outputs
    app6d.ref_eq = app6d.probe_eq = None
    app6d.model_info = None
    app6d.R = app6d.s = app6d.t = None
    app6d.preds = app6d.gt = None
    app6d.probe_goal = None

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
    The orientations (quaternions) are set so that the "front" of each point faces towards the center of its circle.

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
