import numpy as np

from geometry.demos import (load_demo_spirals, load_demo_circles_with_orientation, 
                            load_ref_xyz_from_csv, load_probe_xyz_from_csv
)


def on_press(app6d, event):
    """
    Handle mouse press events for drawing.
    """
    if event.inaxes is None or event.xdata is None or event.ydata is None:
        return

    # Left mouse on XY -> reference
    if event.button == 1 and event.inaxes == app6d.ax_xy:
        app6d.drawing_ref = True
        app6d.ref_legend.set_color("tab:red")
        app6d.line_ref_xy.set_color("tab:red")
        app6d.line_ref_3d.set_color("tab:red")
        app6d.ref_raw = []
        app6d.ref_raw.append(xy_to_xyz(app6d, event))
        app6d.update_ref_lines()
        print("[UI] Start drawing reference (XY)...")

    # Right mouse on YZ -> probe
    if event.button == 3 and event.inaxes == app6d.ax_yz:
        app6d.prediction_id += 1
        app6d.drawing_probe = True
        app6d.probe_raw = []
        app6d.preds = None
        app6d.gt = None
        app6d.probe_raw.append(yz_to_xyz(app6d, event))
        app6d.update_probe_lines()
        app6d.update_pred_lines()
        print("[UI] Start drawing probe (YZ).")

def on_move(app6d, event):
    """
    Handle mouse move events for drawing.
    """
    if event.inaxes is None or event.xdata is None or event.ydata is None:
        return

    if app6d.drawing_ref and event.inaxes == app6d.ax_xy:
        app6d.ref_raw.append(xy_to_xyz(app6d, event))
        app6d.update_ref_lines()

    if app6d.drawing_probe and event.inaxes == app6d.ax_yz:
        app6d.probe_raw.append(yz_to_xyz(app6d, event))
        app6d.update_probe_lines()

def on_release(app6d, event):
    """
    Handle mouse release events to finish drawing.
    """
    # Finish reference
    if event.button == 1 and app6d.drawing_ref:
        app6d.drawing_ref = False
        print(f"[UI] Reference finished. pts={len(app6d.ref_raw)}")
        print()
        return

    # Finish probe -> resample & predict automatically
    if event.button == 3 and app6d.drawing_probe:
        app6d.drawing_probe = False
        print(f"[UI] Probe finished. pts={len(app6d.probe_raw)}")
        print()
        if app6d.use_6d:
            app6d.process_probe_and_predict_6d()
        else:
            app6d.process_probe_and_predict()

def on_key(app6d, event):
    key = event.key

    if key == "c":
        app6d.clear()

    elif key == "h":
        app6d.smooth_enabled = not app6d.smooth_enabled
        print(f"[UI] Smooth enabled: {app6d.smooth_enabled}")
        print()

    elif key == 'L':
        load_demo_spirals(app6d)

    elif key == "O":
        load_demo_circles_with_orientation(app6d)

    elif key == "m":
        app6d.use_6d = not app6d.use_6d

        mode_str = "6D (position + orientation)" if app6d.use_6d else "3D (position only)"
        print(f"[UI] Switched mode -> {mode_str}")
        print()

    elif key == "n":
        app6d.prediction_id += 1
        app6d.ref_raw = app6d.ref_eq = []
        app6d.ref_quat_raw = app6d.ref_quat_eq = None
        app6d.ref_legend.set_color("tab:red")
        app6d.line_ref_xy.set_color("tab:red")
        app6d.line_ref_3d.set_color("tab:red")
        app6d.update_ref_lines()
        print("[UI] Ready to draw a new reference.")
        print()

    elif key == "p":
        if app6d.use_6d:
            app6d.process_probe_and_predict_6d()
        else:
            app6d.process_probe_and_predict()

    elif key == "P":
        load_probe_xyz_from_csv(app6d, "data/ee_trajectory_2026-02-05_17-26-40.csv")

    elif key == "r":
        app6d.prediction_id += 1
        app6d.probe_raw = app6d.probe_eq = []
        app6d.probe_quat_raw = app6d.probe_quat_eq = None
        app6d.preds = app6d.gt = None
        app6d.update_probe_lines()
        app6d.update_pred_lines()
        print("[UI] Probe reset. Ready to draw a new probe.")
        print()

    elif key == "R":
        load_ref_xyz_from_csv(app6d, "data/ee_trajectory_2026-02-05_17-24-52.csv")

    elif key == "t":
        if app6d.use_6d:
            app6d.train_reference_6d()
        else:
            app6d.train_reference()

def xy_to_xyz(app6d, event):
    """
    Convert XY plane event to XYZ coordinate (z=0).
    """
    return np.array([float(event.xdata), float(event.ydata), 0.0], dtype=np.float64)

def yz_to_xyz(app6d, event):
    """
    Convert YZ plane event to XYZ coordinate (x=probe_plane_x).
    """
    return np.array([app6d.probe_plane_x, float(event.xdata), float(event.ydata)], dtype=np.float64)
