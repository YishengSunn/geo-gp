import numpy as np
import traceback

from config.runtime import K_HIST, SAMPLE_HZ, DEFAULT_SPEED, ANCHOR_ANGLE, SELECT_HORIZON
from geometry.angles import angles_relative_to_start_tangent, crossed_multi_in_angle_rel
from viz.plot_traj import plot_series_and_mse
from viz.export_csv import save_csv

def on_press(app, event):
    """
    Mouse button press event handler.
    
    Left button: start drawing reference trajectory
    Right button: start drawing probe trajectory (new prompt)

    Args:
        app: The main application instance.
        event: The mouse event.
    """
    if event.inaxes != app.ax: return

    if event.button == 1:  # Left button: reference
        app.drawing_left = True
        app.ref_pts.append([event.xdata, event.ydata])
        app.update_ref_line()

    elif event.button == 3:  # Right button: target segment (start new prompt)
        # Start a new drawing session: clear probe, reset "within session" counts and sets
        app.drawing_right = True
        app.prediction_id += 1

        # Clear previous prediction
        app.update_scaled_pred([])
        app.update_scaled_pred_smoothed([])

        # Clear previous probe
        app.probe_pts = []
        app.update_probe_line()

        # Add the starting point
        app.probe_pts = [[event.xdata, event.ydata]]
        app.update_probe_line()

        # Initialize last_probe_angle
        app.last_probe_angle = 0.0

        # Clear old t_probe in anchors (to prevent contamination)
        for a in app.anchors:
            if 't_probe' in a:
                del a['t_probe']

        app.current_anchor_ptr = 0  # Reset current anchor pointer

def on_move(app, event):
    """
    Mouse move event handler.

    Args:
        app: The main application instance.
        event: The mouse event.
    """
    if event.inaxes != app.ax: return

    if app.drawing_left:
        app.ref_pts.append([event.xdata, event.ydata])
        app.update_ref_line()

    if app.drawing_right:
        app.probe_pts.append([event.xdata, event.ydata])
        app.update_probe_line()

        # Check crossing of anchors
        app.probe_check_cross_current_anchor()

        # Calculate probe relative angle (relative to probe start tangent)
        probe_np = np.asarray(app.probe_pts, dtype=np.float64)
        if probe_np.shape[0] >= 2:
            probe_rel_angle, mask = angles_relative_to_start_tangent(probe_np, k_hist=K_HIST, min_r=1e-6)
            if mask[-1]:
                th_cur = float(probe_rel_angle[-1])
                app.last_probe_angle = th_cur

    # Only check if all anchors have been crossed in order, then check if the end angle is crossed
    if hasattr(app, 'refs') and app.refs:
        for ref in app.refs:
            # All anchors have been crossed
            if len(ref['probe_crossed_set']) == len(ref['anchors']):
                final_angle = float(ref['anchors'][-1]['angle'])  # Final anchor angle
                if not ref.get('reached_goal', False) and len(app.probe_pts) >= 2:
                    th1, mask = angles_relative_to_start_tangent(app.probe_pts, k_hist=K_HIST, min_r=1e-6)
                    if mask[-1]:
                        th_cur = float(th1[-1])
                        crossed, _ = crossed_multi_in_angle_rel(app.last_probe_angle, th_cur, [final_angle])
                        if crossed:
                            ref['reached_goal'] = True
                            print("All anchors crossed in order, and final angle crossed! Task completed!")

def on_release(app, event):
    """
    Mouse button release event handler.

    Args:
        app: The main application instance.
        event: The mouse event.
    """
    if event.inaxes != app.ax:
        return

    if event.button == 1:
        # Left button release: end reference trajectory drawing
        app.drawing_left = False
        print(f"Reference drawing completed... ")
        print()
        return

    if event.button == 3:
        app.process_probe_and_predict(probe_already_sampled=False)

def on_key(app, event):
    """
    Key press event handler.

    Args:
        app: The main application instance.
        event: The mouse event.
    """
    key = event.key.lower()

    if key == 'a':
        app.show_anchors = not app.show_anchors
        app.draw_anchors()
        print(f"Anchor display: {'ON' if app.show_anchors else 'OFF'} | Total counted={app.anchor_count_total}")
        print()

    elif key == 'b': app.handle_predict_baseline()
    
    elif key == 'c': app.clear_all()

    elif key == 'g': app.process_probe_and_predict(probe_already_sampled=True)

    elif key == 'h':  # Toggle smoothing
        app.smooth_enabled = not app.smooth_enabled
        app.line_smooth.set_visible(app.smooth_enabled)

        if app.smooth_enabled:
            app.line_smooth.set_label("Smoothed Prediction")
        else:
            app.line_smooth.set_label(None)

        handles, labels = app.ax.get_legend_handles_labels()

        if app.smooth_enabled and app.line_smooth not in handles:
            handles.append(app.line_smooth)
            labels.append(app.line_smooth.get_label())

        if not app.smooth_enabled and app.line_smooth in handles:
            idx = handles.index(app.line_smooth)
            handles.pop(idx)
            labels.pop(idx)

        app.ax.legend(handles, labels)
        app.fig.canvas.draw_idle()

        print(f"Smoothing enabled: {app.smooth_enabled}")
        print()

    elif key == 'l': app.load_from_csv("traj_all_20hz_20260111_170954.csv")  # Load from CSV

    elif key == 'm':
        plot_series_and_mse(app.sampled,
                            app.probe_pts,
                            getattr(app, 'pred_scaled', None),
                            getattr(app, 'baseline_preds', None),
                            getattr(app, 'dtheta_manual', 0.0),
                            getattr(app, 'scale_manual', 1.0),
                            SAMPLE_HZ)
        
    elif key == 'n': app.start_new_reference()

    elif key == 'p': app.handle_predict_reference()

    elif key == 'r':  # Reset probe and prediction points and stop prediction
        app.reset_probe_session()
        print("Probe session reset. Ready for new probe.")
        print()

    elif key == 's': save_csv(app)

    elif key == 't': app.handle_train()

    elif key == 'v':
        if app.probe_predict_mode == 'ref-based':
            app.probe_predict_mode = 'probe-based'
        else:
            app.probe_predict_mode = 'ref-based'
        print(f"Current prediction mode switched to: {app.probe_predict_mode}")
        print()

    elif key == 'left': app.move_seed(-1)
    elif key == 'right': app.move_seed(+1)
