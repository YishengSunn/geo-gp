import numpy as np
import traceback

from config.runtime import K_HIST, SAMPLE_HZ, DEFAULT_SPEED, ANCHOR_ANGLE, SELECT_HORIZON
from geometry.angles import angles_relative_to_start_tangent, crossed_multi_in_angle_rel
from geometry.resample import resample_polyline_equal_dt
from matching.align import plot_vectors_at_angle_ref_probe
from matching.ref_selection import choose_best_ref_by_mse
from viz.plot_traj import plot_series_and_mse
from viz.export_csv import save_csv
from utils.misc import closest_index

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

        # Clear previous prediction
        app.update_scaled_pred([])
        app.update_scaled_pred_raw([])

        # Clear previous probe
        app.probe_pts = []
        app.update_probe_line()

        # Add the starting point
        app.probe_pts = [[event.xdata, event.ydata]]
        app.update_probe_line()

        # Initialize last_probe_angle
        app.last_probe_angle = 0.0

        # New session: reset crossing count and crossed set for this session
        app.probe_cross_count_session = 0
        app.probe_crossed_set_session = set()

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
        print("Probe drawing completed, processing probe trajectory...")
        print()
        # Right button release: end probe drawing
        app.drawing_right = False

        # 1) Resample probe with equal time intervals (same as reference trajectory)
        if len(app.probe_pts) >= 2:
            probe_raw = np.asarray(app.probe_pts, dtype=np.float32)

            # Resample with equal time intervals, same as in handle_train for reference trajectory
            probe_eq = resample_polyline_equal_dt(probe_raw, SAMPLE_HZ, DEFAULT_SPEED)

            # Fallback: keep at least two points
            if probe_eq.shape[0] >= 2:
                print(f"Probe resampled with equal time intervals: {len(probe_raw)} → {len(probe_eq)} points")
                app.probe_pts = probe_eq.tolist()
                app.update_probe_line()
        else:
            print("Probe has insufficient points for resampling/matching!")
            # Continue anyway to let subsequent logic provide clearer errors/prompts

        # 2) Select the reference trajectory that best matches the probe (MSE selection)
        if hasattr(app, "refs") and app.refs:
            probe_eq_np  = np.asarray(app.probe_pts, dtype=np.float64)
            probe_raw_np = np.asarray(probe_raw, dtype=np.float64) if 'probe_raw' in locals() else probe_eq_np
            best_idx, best_pack, best_mse = choose_best_ref_by_mse(
                app.refs, probe_eq_np, probe_raw_np, horizon=SELECT_HORIZON, align_on_anchor=False
            )
            print(f"Reference trajectory selection results: best_idx={best_idx}, best_mse={best_mse:.6f}")
            if best_idx is not None:
                app.best_ref = app.refs[best_idx]
                out, dtheta, scale = best_pack
                app.anchor = out
                app.dtheta_manual = dtheta
                app.scale_manual  = scale
                print(f"Best reference #{best_idx} | MSE@{SELECT_HORIZON}={best_mse:.6f} | "
                      f"Δθ={np.degrees(dtheta):.1f}° | s={scale:.3f}")

                # Map original anchor points to "resampled indices" for other visualizations
                ref_resampled   = app.best_ref["sampled"].detach().cpu().numpy()
                probe_resampled = np.asarray(app.probe_pts, dtype=np.float64)
                app.seed_end   = closest_index(app.anchor["ref_point"], ref_resampled)
                app.probe_end  = closest_index(app.anchor["probe_point"], probe_resampled)
            else:
                app.best_ref = None
                print("MSE selection failed: no available reference!")
        else:
            app.best_ref = None
            print("No trained reference trajectories available (refs is empty)!")

        # 3) Visualization: angle change comparison + reference/target vectors
        try:
            if app.best_ref is not None and len(app.probe_pts) > 1:
                ref_np = app.best_ref['sampled'].detach().cpu().numpy()
                probe_np = np.asarray(app.probe_pts, dtype=np.float64)  # Already resampled probe

                # Angle change comparison (relative to start tangent)
                # plot_angle_changes(ref_np, probe_np, k_hist=K_HIST)

                # Target angle vector visualization (example uses 1 rad, can be connected to UI as needed)
                # angle_target = 0.5 
                angle_target = ANCHOR_ANGLE
                out = plot_vectors_at_angle_ref_probe(
                    ref_np, probe_np,
                    angle_target=angle_target,
                    k_hist=K_HIST,
                    n_segments_base=10
                )
                app.seed_end = out['ref_index']
                app.probe_end = out['probe_index']
                v_ref = out['ref_vector']
                v_pro = out['probe_vector']

                app.dtheta_manual = float(np.arctan2(v_pro[1], v_pro[0]) - np.arctan2(v_ref[1], v_ref[0]))
                app.scale_manual = float(np.linalg.norm(v_pro) / max(np.linalg.norm(v_ref), 1e-6))
                print(f"Target vector comparison: Δθ={np.degrees(app.dtheta_manual):.1f}°, scale={app.scale_manual:.3f}")
            else:
                print("Insufficient reference or probe points to plot angle/vector comparison!")
        except Exception as e:
            print(f"Visualization exception: {e}!")
            traceback.print_exc()
        
        # 4) Perform matching and scaling prediction (ref-based / probe-based decided internally)
        try:
            app.match_and_scale_predict()
        except Exception as e:
            print(f"Matching/prediction exception: {e}!")
            traceback.print_exc()

        # 5) Clear all temporary probe states in reference trajectories (prepare for next drawing)
        if hasattr(app, "refs"):
            for ref in app.refs:
                ref['current_anchor_ptr'] = 0
                ref['probe_crossed_set'] = set()
                ref['lookahead_buffer'] = None
                ref['reached_goal'] = False
                for a in ref.get('anchors', []):
                    a.pop('probe_idx', None)
        
        print("Probe processing completed...")
        print()

def on_key(app, event):
    """
    Key press event handler.

    Args:
        app: The main application instance.
        event: The mouse event.
    """
    key = event.key.lower()
    if key == 't': app.handle_train()
    elif key == 'p': app.handle_predict_reference()
    elif key == 'left': app.move_seed(-1)
    elif key == 'right': app.move_seed(+1)
    elif key == 'v':
        if app.probe_predict_mode == 'ref-based':
            app.probe_predict_mode = 'probe-based'
        else:
            app.probe_predict_mode = 'ref-based'
        print(f"Current prediction mode switched to: {app.probe_predict_mode}")
        print()
    elif key == 'c': app.clear_all()
    elif key == 's': save_csv(app)
    elif key == 'g':  # Directly use probe coordinate system for rollout prediction
        app.predict_on_transformed_probe()
    elif key == 'b':
        app.handle_predict_baseline()
    elif key == 'm':
        plot_series_and_mse(app.sampled,
                            app.probe_pts,
                            getattr(app, 'pred_scaled', None),
                            getattr(app, 'baseline_preds', None),
                            getattr(app, 'dtheta_manual', 0.0),
                            getattr(app, 'scale_manual', 1.0),
                            SAMPLE_HZ)
    elif key == 'n': app.start_new_reference()
    elif key == 'a':
        app.show_anchors = not app.show_anchors
        app.draw_anchors()
        print(f"Anchor display: {'ON' if app.show_anchors else 'OFF'} | Total counted={app.anchor_count_total}")
        print()
