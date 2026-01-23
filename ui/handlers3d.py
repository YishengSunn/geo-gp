import numpy as np


def on_press(app3d, event):
    """
    Handle mouse press events for drawing.
    """
    if event.inaxes is None or event.xdata is None or event.ydata is None:
        return

    # Left mouse on XY -> reference
    if event.button == 1 and event.inaxes == app3d.ax_xy:
        app3d.drawing_ref = True
        app3d.ref_raw = []
        app3d.ref_raw.append(xy_to_xyz(app3d, event))
        app3d.update_ref_lines()
        print("[UI] Start drawing reference (XY)...")

    # Right mouse on YZ -> probe
    if event.button == 3 and event.inaxes == app3d.ax_yz:
        app3d.prediction_id += 1
        app3d.drawing_probe = True
        app3d.probe_raw = []
        app3d.preds = None
        app3d.gt = None
        app3d.probe_raw.append(yz_to_xyz(app3d, event))
        app3d.update_probe_lines()
        app3d.update_pred_lines()
        print("[UI] Start drawing probe (YZ).")

def on_move(app3d, event):
    """
    Handle mouse move events for drawing.
    """
    if event.inaxes is None or event.xdata is None or event.ydata is None:
        return

    if app3d.drawing_ref and event.inaxes == app3d.ax_xy:
        app3d.ref_raw.append(xy_to_xyz(app3d, event))
        app3d.update_ref_lines()

    if app3d.drawing_probe and event.inaxes == app3d.ax_yz:
        app3d.probe_raw.append(yz_to_xyz(app3d, event))
        app3d.update_probe_lines()

def on_release(app3d, event):
    """
    Handle mouse release events to finish drawing.
    """
    # Finish reference
    if event.button == 1 and app3d.drawing_ref:
        app3d.drawing_ref = False
        print(f"[UI] Reference finished. pts={len(app3d.ref_raw)}")
        print()
        return

    # Finish probe -> resample & predict automatically
    if event.button == 3 and app3d.drawing_probe:
        app3d.drawing_probe = False
        print(f"[UI] Probe finished. pts={len(app3d.probe_raw)}")
        print()
        app3d.process_probe_and_predict()

def on_key(app3d, event):
    key = event.key.lower()

    if key == "l":
        app3d.load_demo_spirals()

    elif key == "t":
        app3d.train_reference()

    elif key == "p":
        app3d.process_probe_and_predict()

    elif key == "c":
        app3d.clear()

    elif key == "h":
        app3d.smooth_enabled = not app3d.smooth_enabled
        print(f"[UI] Smooth enabled: {app3d.smooth_enabled}")
        print()

    elif key == "n":
        app3d.prediction_id += 1
        app3d.line_ref_xy.set_color("tab:blue")
        app3d.ref_raw = []
        app3d.update_ref_lines()
        print("[UI] Ready to draw a new reference.")
        print()

    elif key == "r":
        app3d.prediction_id += 1
        app3d.probe_raw = []
        app3d.preds = None
        app3d.gt = None
        app3d.update_probe_lines()
        app3d.update_pred_lines()
        print("[UI] Probe reset. Ready to draw a new probe.")
        print()

def xy_to_xyz(app3d, event):
    """
    Convert XY plane event to XYZ coordinate (z=0).
    """
    return np.array([float(event.xdata), float(event.ydata), 0.0], dtype=np.float64)

def yz_to_xyz(app3d, event):
    """
    Convert YZ plane event to XYZ coordinate (x=probe_plane_x).
    """
    return np.array([app3d.probe_plane_x, float(event.xdata), float(event.ydata)], dtype=np.float64)
