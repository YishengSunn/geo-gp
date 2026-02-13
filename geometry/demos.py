import csv
import numpy as np


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
    Load a demo pair of 3D circles with orientations into ref_raw and probe_raw.
    - Reference: small circle on XY plane, oriented to face center
    - Probe: first quarter of larger circle, oriented to face center

    Args:
        T: int, number of points for reference circle
        r_ref: float, radius of reference circle
        r_probe: float, radius of probe circle
    """
    t = np.linspace(0, 2*np.pi, T)

    # Reference small circle on XY plane
    ref_pos = np.stack([r_ref*np.cos(t), r_ref*np.sin(t), np.zeros_like(t)], axis=1)

    # Orientation: x-axis points to center
    ref_rot = []
    for p in ref_pos:
        x_axis = -p / np.linalg.norm(p)
        z_axis = np.array([0, 0, 1.0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        ref_rot.append(R)
    ref_rot = np.asarray(ref_rot)

    # Probe: first quarter of larger circle, translated
    tp = np.linspace(0, np.pi/2, T//4)
    probe_pos = np.stack([np.zeros_like(tp)+1.5, r_probe*np.cos(tp), r_probe*np.sin(tp)], axis=1)

    probe_rot = []
    for p in probe_pos:
        center = np.array([1.5, 0.0, 0.0])
        x_axis = (center - p)
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.array([1.0, 0, 0.0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        probe_rot.append(R)
    probe_rot = np.asarray(probe_rot)

    app6d.ref_raw = ref_pos.tolist()
    app6d.probe_raw = probe_pos.tolist()
    app6d.ref_rot_raw = ref_rot
    app6d.probe_rot_raw = probe_rot

    app6d.update_ref_lines()
    app6d.update_probe_lines()
    app6d.update_pred_lines()

    print(f"[Demo] Loaded circles with orientation: ref_raw={len(app6d.ref_raw)}, probe_raw={len(app6d.probe_raw)}")
    print()

def load_ref_xyz_from_csv(app6d, filepath):
    """
    Load x,y,z columns from a CSV with fields:
    time,x,y,z,qx,qy,qz,qw

    Only xyz are used and written into app6d.ref_raw.

    Args:
        filepath: str, path to the CSV file
    """
    pts = []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"])
                pts.append([x, y, z])
            except Exception:
                continue

    if len(pts) == 0:
        print("[LoadCSV] No valid xyz rows found!")
        print()
        return

    app6d.ref_raw = pts
    app6d.ref_rot_raw = None

    app6d.update_ref_lines()
    print(f"[LoadCSV] Loaded {len(app6d.ref_raw)} reference points from CSV.")
    print()

def load_probe_xyz_from_csv(app6d, filepath):
    """
    Load x,y,z columns from a CSV with fields:
    time,x,y,z,qx,qy,qz,qw

    Only xyz are used and written into app6d.probe_raw.

    Args:
        filepath: str, path to the CSV file
    """
    pts = []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"])
                pts.append([x, y, z])
            except Exception:
                continue

    if len(pts) == 0:
        print("[LoadCSV] No valid xyz rows found!")
        print()
        return

    app6d.probe_raw = pts
    app6d.probe_rot_raw = None

    app6d.update_probe_lines()
    print(f"[LoadCSV] Loaded {len(app6d.probe_raw)} probe points from CSV.")
    print()
