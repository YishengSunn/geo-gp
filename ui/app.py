#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import traceback
import numpy as np
import matplotlib.pyplot as plt

from config.runtime import (
    SEED, TRAIN_RATIO, SAMPLE_HZ, DEFAULT_SPEED, 
    K_HIST, METHOD_ID, METHOD_CONFIGS, 
    MATCH_MODE, MIN_START_ANGLE_DIFF, ANCHOR_ANGLE,
    SELECT_HORIZON
)
from geometry.angles import (
    angle_diff_mod_pi, build_relative_angles, last_window_rel_angles, crossed_multi_in_angle_rel,
)
from geometry.resample import resample_polyline_equal_dt
from geometry.transforms import align_and_scale_gp_prediction
from gp.dataset import build_dataset, time_split
from gp.model import train_gp, gp_predict, rollout_reference
from matching.align import plot_vectors_at_angle_ref_probe
from matching.ref_selection import choose_best_ref_by_mse
from ui.handlers import on_press, on_release, on_move, on_key
from utils.misc import (
    torch_to_np, closest_index, 
    moving_average_centered, smooth_prediction_by_velocity, 
    load_traj_all_csv
)


np.random.seed(SEED)
torch.manual_seed(SEED)


class DrawGPApp:
    def __init__(self):
        """
        Initialize the drawing GUI for GP trajectory prediction.
        """
        # --- States ---
        self.refs = []                   # List of reference trajectories
        self.ref_pts = []                # Reference trajectory points
        self.best_ref = None             # Currently selected reference trajectory
        self.sampled = None              # Sampled trajectory points
        self.probe_pts = []              # Drawn probe points
        self.seed_end = None             # Index of the last seed point in the reference trajectory
        self.probe_end = None            # Index of the last probe point for anchor
        self.model_info = None           # Current GP model info
        self.model_info_baseline = None  # Baseline GP model info
        self.baseline_vars = None        # Variance of baseline rollout
        self.dtheta_manual = 0.0         # Angle offset for manual mode
        self.scale_manual = 1.0          # Scale factor for manual mode
        self.pred_scaled = None          # Scaled predicted points for manual mode
        self.match_mode = MATCH_MODE     # 'angle' or 'manual'
        self.probe_predict_mode = 'probe-based'  # 'probe-based' or 'anchor-based'
        self.rollout_horizon = 2000      # Baseline rollout horizon

        # ---- Anchors / states ----
        self.anchors = []                # List of anchor points
        self.anchor_step = 50            # Steps between anchors
        self.anchor_markers = []         # Anchor marker artists
        self.show_anchors = False        # Whether to show anchor markers
        self.current_anchor_ptr = 0      # Pointer to current anchor
        self.last_probe_angle = 0.0      # Last probe angle for crossing detection
        self.anchor_window = K_HIST      # Window size for angle estimation, used in on_move
        self.ref_rel_angle = None        # Relative angle sequence of reference trajectory
        self.goal_stop_eps = 0.05        # Goal reaching threshold for stopping rollout, used in predict_on_transformed_probe

        # ---- Prediction smoothing ----
        self.smooth_enabled = False       # smooth predicted polyline in probe/world frame
        self.smooth_win = 10             # centered MA window on per-step velocity
        self.smooth_blend_first = 0.8    # keep continuity with probe exiting direction

        # --- Prediction cancellation ID ---
        self.prediction_id = 0           # Incremented on new probe drawing to cancel ongoing predictions

        # --- Prepare a color cycle during class initialization (e.g., 10 colors cycling) ---
        self.past_colors = ["green"] + ["orange"]
        self.past_ref_lines = []         # Historical reference Line2D objects
        self.ref_counter = 0             # Counter: Reference #id

        # --- Global style ---
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.edgecolor": "#000000",
            "axes.linewidth": 1.0,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "legend.frameon": True,
            "legend.edgecolor": "#000000",
            "legend.facecolor": "white",
        })

        # --- GUI Initialization ---
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6))
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect('equal')
        # self.ax.set_title("Trajectory Prediction (Minimal White Style)")

        # --- Background pure white ---
        self.fig.patch.set_facecolor("white")
        self.ax.set_facecolor("white")

        # --- Grid lines (light gray, thin) ---
        self.ax.grid(True, color="#DDDDDD", linestyle="--", linewidth=0.6, alpha=0.8)
        
        # --- Curves to display ---
        self.line_ref, = self.ax.plot([], [], '-', color='red', lw=3.0, label='Demonstration')  # Demo trajectory (reference)
        self.line_probe, = self.ax.plot([], [], '-', color='black', lw=3.0, label='Prompt')  # Probe trajectory
        self.line_ps, = self.ax.plot([], [], '-', color='blue', lw=3.0, label='Prediction')  # Predicted trajectory
        self.line_smooth, = self.ax.plot([], [], '-', color='orange', lw=3.0, visible=self.smooth_enabled, alpha=0.75)  # Smoothed predicted trajectory
        self.line_samp, = self.ax.plot([], [], '.', color='#FF7F0E', markersize=2, visible=False)
        self.line_seed, = self.ax.plot([], [], '-', color='black', lw=1.5, visible=False)
        self.line_gt, = self.ax.plot([], [], '-', color='purple', lw=1.0, visible=False)
        self.line_pred, = self.ax.plot([], [], '-', color='green', lw=1.0, visible=False)

        # -- Legend ---
        self.ax.legend(fontsize=14, loc='upper right')

        # --- Events ---
        self.drawing_left = False
        self.drawing_right = False
        self.cid_press   = self.fig.canvas.mpl_connect('button_press_event', lambda event: on_press(self, event))
        self.cid_move    = self.fig.canvas.mpl_connect('motion_notify_event', lambda event: on_move(self, event))
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', lambda event: on_release(self, event))
        self.cid_key     = self.fig.canvas.mpl_connect('key_press_event', lambda event: on_key(self, event))

        plt.tight_layout()
        plt.show(block=True)

    # ============================================================
    # Handlers for Training and Prediction
    # ============================================================

    # Train GP model on drawn reference trajectory
    def handle_train(self):
        """
        Train a GP model on the currently drawn reference trajectory.
        """
        print("Training started...")

        if len(self.ref_pts) < 2:
            print("Please draw a reference trajectory with the left mouse button (at least 2 points)!")
            print("Training aborted...")
            print()
            return

        sampled = resample_polyline_equal_dt(self.ref_pts, SAMPLE_HZ, DEFAULT_SPEED)  # Shape: (N, 2)

        if self.smooth_enabled:
            sampled = moving_average_centered(sampled, win=self.smooth_win)

        if sampled.shape[0] < K_HIST + 2:
            print(f"Too few samples {sampled.shape[0]} < {K_HIST+2}!")
            print("Training aborted...")
            print()
            return
        self.sampled = torch.tensor(sampled, dtype=torch.float32)
        
        input_type, output_type = METHOD_CONFIGS[METHOD_ID]
        X, Y = build_dataset(self.sampled, K_HIST, input_type, output_type)
        (Xtr, Ytr), (Xte, Yte), ntr = time_split(X, Y, TRAIN_RATIO)
        ds = {'X_train': Xtr, 'Y_train': Ytr, 'X_test': Xte, 'Y_test': Yte, 'n_train': ntr}
        self.model_info = train_gp(ds, METHOD_ID)

        # self.seed_end = max(K_HIST-1, min(self.sampled.shape[0]-2, int(self.sampled.shape[0]*0.33)))
        self.update_sample_line(); self.update_seed_line(); self.update_ref_pred_gt(None, None)

        # If there is an existing current reference line, transfer it to "historical references"
        color_idx = self.ref_counter % len(self.past_colors)
        past_color = self.past_colors[color_idx]
        self.ref_counter += 1
        self.line_ref.set_data(self.sampled[:,0], self.sampled[:,1])
        self.line_ref.set_zorder(1)
        self.line_ref.set_linewidth(3.0)
        self.line_ref.set_color(past_color)
        self.line_ref.set_label(f"Demonstration #{self.ref_counter}")
        self.past_ref_lines.append(self.line_ref)

        # Create a new line_ref for the current reference to avoid overwriting past references
        x_new, y_new = [], []
        self.line_ref, = self.ax.plot(
            x_new, y_new, color="red", linewidth=3.0, label="Current Demonstration"
        )
        
        # Do not remove: keep on canvas
        self.ax.legend(fontsize=14, loc='upper right')
        self.fig.canvas.draw_idle()
        
        # Build relative angle anchors (every anchor_step points)
        ref_np = self.sampled.numpy()
        self.ref_rel_angle = build_relative_angles(ref_np, k_hist=K_HIST, origin_idx=0, min_r=1e-6)

        self.anchors = []
        anchor_indices = []
        step = max(1, int(self.anchor_step))

        for i in range(step, len(self.ref_rel_angle), step):
            angle = self.ref_rel_angle[i]
            if np.isnan(angle):
                continue
            if len(anchor_indices) == 0:
                # The first anchor point: angle difference with the start point must be greater than the threshold
                angle_diff = abs(angle_diff_mod_pi(angle, 0.0))  # Relative to the start direction
                if angle_diff >= MIN_START_ANGLE_DIFF:
                    anchor_indices.append(i)
            else:
                anchor_indices.append(i)

        # Make sure the last point is an anchor
        if (len(self.ref_rel_angle) - 1) not in anchor_indices:
            anchor_indices.append(len(self.ref_rel_angle) - 1)

        # Create anchors
        self.anchors = []
        for i in anchor_indices:
            self.anchors.append({
                'idx': i,
                'angle': float(self.ref_rel_angle[i]),
                't_ref': i / SAMPLE_HZ
            })

        # Make sure the last point is an anchor
        if (len(self.ref_rel_angle) - 1) not in [a['idx'] for a in self.anchors]:
            j = len(self.ref_rel_angle) - 1
            self.anchors.append({'idx': j, 'angle': float(self.ref_rel_angle[j]), 't_ref': j / SAMPLE_HZ})

        # Remove the first point as an anchor
        if self.anchors and self.anchors[0]['idx'] == 0:
            self.anchors = self.anchors[1:]
    
        self.draw_anchors()
        self.current_anchor_ptr = 0
        print(f"(relative) Fixed anchors generated: {len(self.anchors)} (step={self.anchor_step})")

        # Save the trained reference trajectory to self.refs
        self.refs.append(dict(
            sampled=self.sampled,
            model_info=self.model_info,
            anchors=[dict(a) for a in self.anchors],
            current_anchor_ptr=0,
            probe_crossed_set=set(),
            lookahead_buffer=None,
            reached_goal=False,
            raw=np.array(self.ref_pts, dtype=np.float32)  # Added: save the original reference trajectory
        ))
        print(f"Total trained references: {len(self.refs)}")
        
        # Clear reference trajectory after training
        self.ref_pts.clear()
        self.line_ref.set_data([], [])
        self.fig.canvas.draw_idle()
        print("All reference trajectories hidden after training")
        print("Training completed...")
        print()

    # Reference trajectory rollout prediction
    def handle_predict_reference(self):
        """
        Perform reference trajectory rollout prediction based on the current seed_end.
        Update the reference prediction and ground truth lines.
        Calculate and print MSE if ground truth is available.
        """
        print("Reference rollout prediction started...")

        if self.model_info is None or self.sampled is None:
            print("Please train first (Press T)!")
            print("Reference rollout prediction aborted...")
            print()
            return
        
        if self.seed_end is None:
            self.seed_end = K_HIST - 1
            
        input_type, output_type = METHOD_CONFIGS[METHOD_ID]
        start_t = int(self.seed_end)
        h = self.sampled.shape[0] - (start_t + 1)
        preds, gt, h_used = rollout_reference(self.model_info, self.sampled, start_t, h, K_HIST, input_type, output_type)

        preds_np = preds.numpy() if preds.numel() > 0 else np.zeros((0, 2), dtype=np.float32)
        gt_np = gt.numpy() if gt.numel() > 0 else np.zeros((0, 2), dtype=np.float32)

        self.update_ref_pred_gt(preds_np, gt_np)
        mse = float(((preds - gt)**2).mean().item()) if gt.numel() > 0 else float('nan')
        print(f"Reference Prediction: h={h_used} | MSE={mse:.6f}")
        print("Reference rollout prediction completed...")
        print()

    # Probe-based rollout prediction using baseline GP model
    def handle_predict_baseline(self):
        """
        Use the baseline GP model to rollout from the probe points.
        Press button 'b' to trigger.
        """
        print("Probe-based rollout prediction using baseline GP model started...")

        if not hasattr(self, "model_info_baseline") or self.model_info_baseline is None:
            print("Baseline GP model not found (please train first)!")
            print("Probe-based rollout prediction using baseline GP model aborted...")
            print()
            return
        
        if self.probe_pts is None or len(self.probe_pts) < K_HIST:
            print(f"Probe too short, at least {K_HIST} points required!")
            print("Probe-based rollout prediction using baseline GP model aborted...")
            print()
            return

        # --- Step 1: probe → ref frame---
        probe_np = np.asarray(self.probe_pts, dtype=np.float64)

        # --- Step 2: baseline rollout in probe frame ---
        cur_hist = [torch.as_tensor(pt, dtype=torch.float32) for pt in probe_np[-K_HIST:]]
        preds_ref, vars_ref = [], []
        rollout_horizon = 500

        for _ in range(rollout_horizon):
            X = torch.cat(cur_hist[-K_HIST:]).reshape(1, -1)  # Shape: (1, K_HIST*2)
            y_pred, var_pred = gp_predict(self.model_info_baseline, X)  # mu: (1, 2), var: (1, 2)
            mu = torch.as_tensor(y_pred[0], dtype=torch.float32)
            preds_ref.append(mu)
            vars_ref.append(var_pred)
            cur_hist.append(mu)

        preds_ref = torch.stack(preds_ref, dim=0).numpy()  # (H, 2)
        vars_ref = np.array(vars_ref)  # (H, 2)

        # --- Step 3: goal = last point of reference trajectory ---
        if self.sampled is not None and preds_ref.shape[0] > 0:
            ref_goal = torch_to_np(self.sampled[-1])  # Reference goal
            dists = np.linalg.norm(preds_ref - ref_goal[None, :], axis=1)  # Shape: (H,)
            hits = np.where(dists <= self.goal_stop_eps)[0]

            for cut_idx in hits:
                var_at_hit = np.max(vars_ref[cut_idx])
                print(f"[Baseline] idx={cut_idx} | d={dists[cut_idx]:.3f} | var={var_at_hit:.3f}")
                if var_at_hit > 0.001:  # ✅ Both thresholds met
                    print(f"Baseline truncation: hit ref goal threshold and var > 0.001 → cut_idx={cut_idx}")
                    preds_ref = preds_ref[:cut_idx]
                    vars_ref = vars_ref[:cut_idx]
                    break

        self.baseline_preds = preds_ref   # ✅ Save baseline rollout trajectory
        self.baseline_vars = vars_ref     # ✅ Save variance of baseline rollout

        print("Probe-based rollout prediction using baseline GP model completed...")
        print()

    # Probe-based rollout prediction using SkyGP model
    def predict_on_transformed_probe(self):
        """
        Perform probe-based dynamic rollout prediction.
        Press button 'p' to trigger.
        """
        print("Probe-based rollout prediction using SkyGP model started...")

        self.prediction_id += 1
        local_pred_id = self.prediction_id

        if not hasattr(self, "best_ref") or self.best_ref is None:
            print("Best reference trajectory not found (please draw probe first)!")
            print("Probe-based rollout prediction using SkyGP model aborted...")
            print()
            return
        
        if len(self.probe_pts) < K_HIST:
            print(f"Probe too short, at least {K_HIST} points required!")
            print("Probe-based rollout prediction using SkyGP model aborted...")
            print()
            return

        # Step 0: data preparation
        ref_np = self.best_ref['sampled'].numpy()
        model_info = self.best_ref['model_info']
        probe_np = np.asarray(self.probe_pts, dtype=np.float64)

        # Step 1: Δθ and scale
        dtheta = self.dtheta_manual
        spatial_scale = self.scale_manual
        print(f"Using manual mode for Δθ and scale: Δθ={np.degrees(dtheta):.2f}°, scale={spatial_scale:.3f}")

        # Step 2: probe → ref frame
        c, s = np.cos(-dtheta), np.sin(-dtheta)
        R_inv = np.array([[c, -s], [s, c]])
        probe_origin = probe_np[0]
        probe_in_ref = ((probe_np - probe_origin) @ R_inv.T) / spatial_scale

        # Step 3: goal (ref last point → probe frame)
        c_f, s_f = np.cos(dtheta), np.sin(dtheta)
        R_fwd = np.array([[c_f, -s_f], [s_f, c_f]])
        ref_vec_total = ref_np[-1] - ref_np[0]
        probe_goal = probe_origin + spatial_scale * np.dot(ref_vec_total, R_fwd.T)
        self.probe_goal = probe_goal

        # Step 4: point-by-point rollout (h = 1 per step)
        self.pred_scaled = []  # Unsmoothed prediction points in world/probe frame
        self.pred_smooth = []  # Smoothed prediction points in world/probe frame
        cur_hist = probe_in_ref.copy()  # Current history trajectory (in ref frame)
        input_type, output_type = METHOD_CONFIGS[METHOD_ID]
        rollout_horizon = self.rollout_horizon

        for step in range(rollout_horizon):
            if local_pred_id != self.prediction_id:
                self.update_scaled_pred()
                print("Prediction cancelled by new prompt.")
                print()
                return

            preds_ref, _, _, vars_ref = rollout_reference(
                model_info,
                torch.tensor(cur_hist, dtype=torch.float32),
                start_t=cur_hist.shape[0] - 1,
                h=1,  # Rollout one step at a time
                k=K_HIST,
                input_type=input_type,
                output_type=output_type
            )

            next_ref = preds_ref[-1].numpy()                     # Take the last predicted point
            next_pos_world = probe_origin + spatial_scale * np.dot(next_ref, R_fwd.T)  # Transform back to probe frame
            self.pred_scaled.append(next_pos_world)              # Store unsmoothed prediction

            if step % 10 == 0:
                self.update_scaled_pred(np.array(self.pred_scaled))
                plt.pause(0.001)

            cur_hist = np.vstack([cur_hist, next_ref])           # Update history

            # Truncation logic
            if self.probe_goal is not None:
                d = np.linalg.norm(next_pos_world - self.probe_goal)
                if d <= self.goal_stop_eps and np.max(vars_ref) > 0.001:
                    print(f"Probe-based truncation: step={step}, d={d:.3f}, var={np.max(vars_ref):.3f}")
                    break

        # Step 5: post-process: optional smoothing (display only, no feedback to GP)
        if self.smooth_enabled and len(self.pred_scaled) >= 2:
            pred_raw_arr = np.asarray(self.pred_scaled, dtype=np.float64)
            pred_smooth = smooth_prediction_by_velocity(
                probe=probe_np,
                pred=pred_raw_arr,
                win=self.smooth_win,
                blend_first_step=self.smooth_blend_first,
            )
            self.update_scaled_pred_smoothed(pred_smooth)
        else:
            self.update_scaled_pred_smoothed([])

        print("Probe-based rollout prediction using SkyGP model completed...")
        print()
        
    # Matching & scaling prediction (including relative angle anchor counting + local seed search)
    def match_and_scale_predict(self):
        """
        Perform matching and scaling prediction based on the current probe trajectory.
        Two modes:
        1) Reference trajectory rollout + mapping (default)
        2) Direct probe-based prediction (if self.probe_predict_mode == 'probe-based')
        3) Update the scaled prediction line and print relevant information.
        4) Calculate and print MSE if ground truth is available.

        Two prediction modes:
        - ref-based: rollout on the reference trajectory, then map to the probe coordinate system
        - probe-based: directly use the probe's seed rollout, independent of the reference trajectory
        """
        print("Matching and scaling prediction started...")
        print()

        if self.best_ref is None:
            print("Please train first (T)!")
            print("Matching and scaling prediction aborted...")
            print()
            return

        input_type, output_type = METHOD_CONFIGS[METHOD_ID]

        # Mode A: Transform probe to reference coordinate system, predict and transform back
        if self.probe_predict_mode == 'probe-based':
            self.predict_on_transformed_probe()
            print("Matching and scaling prediction completed...")
            print()
            return

        # Mode B: Reference trajectory rollout + mapping
        if self.sampled is None or self.seed_end is None:
            print("Missing reference trajectory or seed_end!")
            print("Matching and scaling prediction aborted...")
            print()
            return

        if len(self.probe_pts) < 2:
            print("Probe too short!")
            print("Matching and scaling prediction aborted...")
            print()
            return

        try:
            start_t = int(self.seed_end)
            h = self.sampled.shape[0] - (start_t + 1)
            preds_ref, gt_ref, h_used, _ = rollout_reference(
                self.model_info, self.sampled, start_t, h, K_HIST, input_type, output_type
            )
        except Exception as e:
            print(f"Reference rollout failed: {e}!")
            print("Matching and scaling prediction aborted...")
            print()
            return

        preds_ref_np = preds_ref.numpy() if preds_ref is not None and preds_ref.numel() > 0 else np.zeros((0,2), dtype=np.float32)
        ref_traj_np = self.sampled.numpy()

        try:
            preds_tar, params = align_and_scale_gp_prediction(
                ref_traj_np=ref_traj_np,
                seed_end=self.seed_end,
                probe_end=self.probe_end,
                K_hist=K_HIST,
                preds_ref_np=preds_ref_np,
                probe_points=self.probe_pts,
                mode=self.match_mode
            )
        except Exception as e:
            print(f"Matching failed: {e}!")
            print("Matching and scaling prediction aborted...")
            print()
            return

        self.update_scaled_pred(preds_tar)

        if gt_ref is not None and gt_ref.numel() > 0:
            mse_ref = float(((preds_ref - gt_ref)**2).mean().item())
            pretty = {k: (np.round(v, 4) if isinstance(v, np.ndarray) else v) for k, v in params.items()}
            print(f"Ref-based matching and scaling prediction results: "
                  f"Mode={self.match_mode} | seed_end={self.seed_end} | MSE={mse_ref:.6f} | Params: {pretty}")
            print("Matching and scaling prediction completed...")
        else:
            print("Matching and scaling prediction completed...")

        print()

    def process_probe_and_predict(self, *, probe_already_sampled=False):
        """
        Process the drawn probe trajectory:
        1) Resample probe with equal time intervals
        2) Select the best matching reference trajectory (MSE selection)
        3) Visualization: angle change comparison + reference/target vectors
        4) Perform matching and scaling prediction
        5) Clear temporary probe states in reference trajectories
        6) Update the scaled prediction line and print relevant information
        7) Calculate and print MSE if ground truth is available
        8) If probe_already_sampled is True, skip resampling step

        Args:
            probe_already_sampled (bool): If True, skip the resampling step.
        """
        print("Probe drawing completed, processing probe trajectory...")
        print()

        # Right button release: end probe drawing
        if self.drawing_right:
            self.drawing_right = False

        # 1) Resample probe with equal time intervals (same as reference trajectory)
        if len(self.probe_pts) >= 2:
            probe_raw = np.asarray(self.probe_pts, dtype=np.float32)

            # Resample with equal time intervals, same as in handle_train for reference trajectory
            probe_eq = resample_polyline_equal_dt(probe_raw, SAMPLE_HZ, DEFAULT_SPEED) if not probe_already_sampled else probe_raw

            if self.smooth_enabled:
                probe_eq = moving_average_centered(probe_eq, win=self.smooth_win)
                self.update_probe_line()

            # Fallback: keep at least two points
            if probe_eq.shape[0] >= 2:
                print(f"Probe resampled with equal time intervals: {len(probe_raw)} → {len(probe_eq)} points")
                self.probe_pts = probe_eq.tolist()
                self.update_probe_line()
        else:
            print("Probe has insufficient points for resampling/matching!")
            # Continue anyway to let subsequent logic provide clearer errors/prompts

        # 2) Select the reference trajectory that best matches the probe (MSE selection)
        if hasattr(self, "refs") and self.refs:
            probe_eq_np  = np.asarray(self.probe_pts, dtype=np.float64)
            probe_raw_np = np.asarray(probe_raw, dtype=np.float64) if 'probe_raw' in locals() else probe_eq_np
            best_idx, best_pack, best_mse = choose_best_ref_by_mse(
                self.refs, probe_eq_np, probe_raw_np, horizon=SELECT_HORIZON, align_on_anchor=False
            )
            print(f"Reference trajectory selection results: best_idx={best_idx}, best_mse={best_mse:.6f}")
            if best_idx is not None:
                self.best_ref = self.refs[best_idx]
                out, dtheta, scale = best_pack
                self.anchor = out
                self.dtheta_manual = dtheta
                self.scale_manual  = scale
                print(f"Best reference #{best_idx} | MSE@{SELECT_HORIZON}={best_mse:.6f} | "
                      f"Δθ={np.degrees(dtheta):.1f}° | s={scale:.3f}")

                # Map original anchor points to "resampled indices" for other visualizations
                ref_resampled   = self.best_ref["sampled"].detach().cpu().numpy()
                probe_resampled = np.asarray(self.probe_pts, dtype=np.float64)
                self.seed_end   = closest_index(self.anchor["ref_point"], ref_resampled)
                self.probe_end  = closest_index(self.anchor["probe_point"], probe_resampled)
            else:
                self.best_ref = None
                print("MSE selection failed: no available reference!")
        else:
            self.best_ref = None
            print("No trained reference trajectories available (refs is empty)!")

        # 3) Visualization: angle change comparison + reference/target vectors
        try:
            if self.best_ref is not None and len(self.probe_pts) > 1:
                ref_np = self.best_ref['sampled'].detach().cpu().numpy()
                probe_np = np.asarray(self.probe_pts, dtype=np.float64)  # Already resampled probe

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
                self.seed_end = out['ref_index']
                self.probe_end = out['probe_index']
                v_ref = out['ref_vector']
                v_pro = out['probe_vector']

                self.dtheta_manual = float(np.arctan2(v_pro[1], v_pro[0]) - np.arctan2(v_ref[1], v_ref[0]))
                self.scale_manual = float(np.linalg.norm(v_pro) / max(np.linalg.norm(v_ref), 1e-6))
                print(f"Target vector comparison: Δθ={np.degrees(self.dtheta_manual):.1f}°, scale={self.scale_manual:.3f}")
            else:
                print("Insufficient reference or probe points to plot angle/vector comparison!")
        except Exception as e:
            print(f"Visualization exception: {e}!")
            traceback.print_exc()
        
        # 4) Perform matching and scaling prediction (ref-based / probe-based decided internally)
        try:
            self.match_and_scale_predict()
        except Exception as e:
            print(f"Matching/prediction exception: {e}!")
            traceback.print_exc()

        # 5) Clear all temporary probe states in reference trajectories (prepare for next drawing)
        if hasattr(self, "refs"):
            for ref in self.refs:
                ref['current_anchor_ptr'] = 0
                ref['probe_crossed_set'] = set()
                ref['lookahead_buffer'] = None
                ref['reached_goal'] = False
                for a in ref.get('anchors', []):
                    a.pop('probe_idx', None)
        
        print("Probe processing completed...")
        print()

    # ============================================================
    # Handlers for UI Actions
    # ============================================================

    def load_from_csv(self, path):
        """
        Load reference and probe trajectories from CSV files.
        
        Args:
            path (str): Path to the CSV file containing trajectory data.
        """
        print(f"Loading trajectories from CSV: {path}...")

        if (not path.endswith('.csv')):
            print("Please provide a valid CSV file path!")
            print("Loading aborted...")
            print()
            return
        
        self.ax.set_xscale("linear")
        self.ax.set_yscale("linear")

        ref, sampled, probe = load_traj_all_csv(path)

        # Reset states
        self.reset_probe_session()
        self.ref_pts.clear()
        self.probe_pts.clear()
        self.sampled = None

        # Load data
        if len(ref) > 0:
            self.ref_pts = ref.tolist()

        if len(probe) > 0:
            self.probe_pts = probe.tolist()

        if len(sampled) > 0:
            self.sampled = torch.tensor(sampled, dtype=torch.float32)

        self.update_ref_line()
        self.update_probe_line()
        self.fig.canvas.draw_idle()

        print("Trajectories loaded from CSV...")
        print(
            f"Loaded CSV:\n"
            f"  ref_pts   = {len(ref)}\n"
            f"  sampled   = {len(sampled)}\n"
            f"  probe_pts = {len(probe)}"
        )
        print()

    # Start a new reference trajectory
    def start_new_reference(self):
        """
        Start a new reference: clear the current reference's temporary state, keep trained refs.
        """
        self.ref_pts = []
        self.sampled = None
        self.model_info = None
        self.seed_end = None

        self.anchors = []
        self.ref_rel_angle = None
        self.current_anchor_ptr = 0

        self.line_ref.set_data([], [])
        self.fig.canvas.draw_idle()
        print("Started a new reference trajectory (previous trained references kept).")
        print()

    def reset_probe_session(self):
        """
        Reset the current probe session: clear probe points and related states.
        """
        self.prediction_id += 1

        self.update_scaled_pred([])
        self.update_scaled_pred_smoothed([])

        self.probe_pts.clear()
        self.update_probe_line()

        self.last_probe_angle = 0.0
        self.current_anchor_ptr = 0
        self.best_ref = None

        for ref in self.refs:
            ref['current_anchor_ptr'] = 0
            ref['probe_crossed_set'] = set()
            ref['lookahead_buffer'] = None
            ref['reached_goal'] = False
            for a in ref.get('anchors', []):
                a.pop('probe_idx', None)
                a.pop('t_probe', None)

        self.drawing_right = False
        self.drawing_left = False

    # Clear all data and visualization
    def clear_all(self):
        """
        Clear all data and visualization.
        """
        print("Clearing all data and visualization...")
        
        self.refs.clear()
        self.ref_pts.clear()
        self.probe_pts.clear()
        self.sampled = None
        self.model_info = None
        self.seed_end = None
        self.probe_end = None

        # —— Clear anchor data and visualization ——
        self.anchors = []
        self.ref_rel_angle = None
        for h in getattr(self, "anchor_markers", []):
            try:
                h.remove()
            except Exception:
                pass
        self.anchor_markers.clear()
        self.current_anchor_ptr = 0
        
        if getattr(self, "h_goal", None) is not None:
            try:
                self.h_goal.remove()
            except Exception:
                pass
            self.h_goal = None
        self.probe_goal = None

        if self.line_ref:
            self.line_ref.set_data([], [])
        self.update_probe_line()
        self.update_sample_line()
        self.update_scaled_pred(None)
        self.update_scaled_pred_smoothed(None)
        self.update_ref_pred_gt(None, None)
        self.update_seed_line()

        # Clear historical reference lines
        if hasattr(self, "past_ref_lines"):
            for ln in self.past_ref_lines:
                try:
                    ln.remove()
                except Exception:
                    pass
            self.past_ref_lines = []
        self.ref_counter = 0

        self.line_smooth.set_visible(self.smooth_enabled)
        self.ax.legend()
        self.fig.canvas.draw_idle()

        print("All data and visualization cleared.")
        print()

    # Seed Adjustment
    def move_seed(self, delta):
        """
        Move the seed_end index by delta steps.

        Args:
            delta (int): Number of steps to move the seed_end index.
        """
        print(f"Moving seed_end by {delta} steps...")

        if self.sampled is None:
            print("Please train first (T)!")
            print("Move seed_end aborted...")
            print()
            return
        
        new_end = (self.seed_end if self.seed_end is not None else (K_HIST - 1)) + int(delta)
        self.seed_end = max(K_HIST-1, min(self.sampled.shape[0]-2, new_end))  # K_HIST-1 ≤ seed_end ≤ len(sampled)-2
        self.update_seed_line()
        print(f"Seed_end={self.seed_end}")
        print("Move seed_end completed...")
        print()
            
    # ============================================================
    # Anchor Visualization
    # ============================================================

    # Draw anchor markers of the reference trajectory
    def draw_anchors(self):
        """
        Draw anchor markers on the plot.
        """
        for h in self.anchor_markers:
            try:
                h.remove()
            except Exception:
                pass
        self.anchor_markers.clear()

        if not self.show_anchors or self.sampled is None or not self.anchors:
            self.fig.canvas.draw_idle()
            return

        ref_np = self.sampled.numpy()
        for k, a in enumerate(self.anchors):
            i = a['idx']
            if 0 <= i < len(ref_np):
                p = ref_np[i]
                m = self.ax.scatter(p[0], p[1], s=20, marker='o', color='black', zorder=4)
                txt = self.ax.text(
                    p[0], p[1],
                    f"A{k}", fontsize=7, color='black',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='black', alpha=0.6),
                    zorder=5
                )
                self.anchor_markers.extend([m, txt])
        self.fig.canvas.draw_idle()

    # Draw anchor markers of the probe trajectory
    def draw_probe_anchors(self):
        """
        Draw markers for probe anchors based on the current probe trajectory and anchors.
        """
        # Remove old ones
        for h in self.probe_anchor_markers:
            try:
                h.remove()
            except Exception:
                pass
        self.probe_anchor_markers.clear()

        if len(self.probe_pts) < 1:
            self.fig.canvas.draw_idle()
            return

        pts = np.asarray(self.probe_pts, dtype=np.float64)
        for k, a in enumerate(self.anchors):
            if 't_probe' in a:
                t_probe = a['t_probe']
                idx = int(round(t_probe * SAMPLE_HZ))
                if 0 <= idx < len(pts):
                    p = pts[idx]
                    m = self.ax.scatter(p[0], p[1], s=20, marker='x', color='blue', zorder=5)
                    txt = self.ax.text(
                        p[0], p[1],
                        f"P{k}", fontsize=7, color='blue',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='blue', alpha=0.6),
                        zorder=6
                    )
                    self.probe_anchor_markers.extend([m, txt])

        self.fig.canvas.draw_idle()

    # ============================================================
    # Anchor Crossing Check
    # ============================================================

    # Check if the current probe crosses reference trajectory anchors
    def probe_check_cross_current_anchor(self):
        """
        Check if the current probe crosses the anchors of each reference trajectory.
        Each reference trajectory independently maintains its current_anchor_ptr, probe_crossed_set, and lookahead_buffer.

        Returns:
            changed_refs: int, number of reference trajectories with crossing actions (for counting only)
        """
        if len(self.probe_pts) < 2 or not self.refs:
            return 0

        th0 = self.last_probe_angle  # Previous probe angle
        th1, mask = last_window_rel_angles(self.probe_pts, W=self.anchor_window, min_r=1e-3)
        if th1 is None or not mask[-1]:
            return 0

        changed_refs = 0  # Number of trajectories with crossing actions (for counting only)
        cur_probe_idx = len(self.probe_pts) - 1

        for ref_idx, ref in enumerate(self.refs):
            anchors = ref['anchors']  # List of anchor dicts, each with 'angle' and 'idx'
            ptr = ref['current_anchor_ptr']
            buffer = ref.get('lookahead_buffer', None)

            idx0, idx1, idx2 = ptr, ptr + 1, ptr + 2

            def get_angle(i):
                """
                Get anchor angle by index, or None if out of range.

                Returns:
                    angle (float) or None
                """
                return anchors[i]['angle'] if i < len(anchors) else None

            crossed0 = crossed_multi_in_angle_rel(th0, th1[-1], [get_angle(idx0)])[0] if idx0 < len(anchors) else False
            crossed1 = crossed_multi_in_angle_rel(th0, th1[-1], [get_angle(idx1)])[0] if idx1 < len(anchors) else False
            crossed2 = crossed_multi_in_angle_rel(th0, th1[-1], [get_angle(idx2)])[0] if idx2 < len(anchors) else False

            # === 1. Cross current anchor idx0 ===
            if crossed0:
                ref['probe_crossed_set'].add(idx0)
                ref['current_anchor_ptr'] = idx0 + 1
                ref['lookahead_buffer'] = None
                anchors[idx0]['probe_idx'] = cur_probe_idx
                changed_refs += 1
                print(f"[AnchorCross] Ref {ref_idx}: crossed A{idx0} at probe_idx={cur_probe_idx} → advance ptr to {idx0 + 1}")
                continue

            # === 2. Cross next anchor idx1, set lookahead buffer ===
            elif crossed1:
                ref['lookahead_buffer'] = {
                    'anchor_idx': idx1,
                    'probe_idx': cur_probe_idx
                }
                print(f"[AnchorLookahead] Ref {ref_idx}: crossed A{idx1} at probe_idx={cur_probe_idx}, buffering…")

            # === 3. Lookahead buffer set, check if crossing idx2 consecutively ===
            if buffer and crossed2:
                k1 = buffer['anchor_idx']
                k2 = idx2

                if 0 <= k1 < len(anchors) and k1 not in ref['probe_crossed_set']:
                    ref['probe_crossed_set'].add(k1)
                    anchors[k1]['probe_idx'] = buffer['probe_idx']
                if 0 <= k2 < len(anchors) and k2 not in ref['probe_crossed_set']:
                    ref['probe_crossed_set'].add(k2)
                    anchors[k2]['probe_idx'] = cur_probe_idx

                print(
                    f"[AnchorConfirm] Ref {ref_idx}: confirm lookahead A{k1} and A{k2} "
                    f"(probe_idx={buffer['probe_idx']}→{cur_probe_idx}), advance ptr to {k2 + 1}"
                )

                ref['current_anchor_ptr'] = k2 + 1
                ref['lookahead_buffer'] = None
                changed_refs += 1

        return changed_refs

    # ============================================================
    # Visualization Update Functions
    # ============================================================

    def update_ref_line(self):
        """
        Update the reference trajectory line based on current ref_pts.
        """
        if self.ref_pts:
            pts = np.asarray(self.ref_pts, dtype=np.float32)
            self.line_ref.set_data(pts[:, 0], pts[:, 1])
        else:
            self.line_ref.set_data([], [])
        self.fig.canvas.draw_idle()

    def update_probe_line(self):
        """
        Update the probe trajectory line based on current probe_pts.
        """
        if self.probe_pts:
            pts = np.asarray(self.probe_pts, dtype=np.float32)
            self.line_probe.set_data(pts[:,0], pts[:,1])
        else:
            self.line_probe.set_data([], [])
        self.fig.canvas.draw_idle()

    def update_scaled_pred(self, preds_scaled=None):
        """
        Update the scaled prediction trajectory line based on current preds_scaled.

        Args:
            preds_scaled (list or np.ndarray): List or array of predicted points in world frame.
        """
        if preds_scaled is not None and len(preds_scaled) > 0:
            arr = np.array(preds_scaled)  # Convert to numpy for plotting
            self.line_ps.set_data(arr[:, 0], arr[:, 1])
            self.pred_scaled = list(map(tuple, preds_scaled))  # Store as list to ensure appendable
        
        else:
            self.line_ps.set_data([], [])
            self.pred_scaled = []
        self.fig.canvas.draw_idle()

    def update_scaled_pred_smoothed(self, preds_scaled_smooth=None):
        """
        Update the smoothed scaled prediction trajectory line based on current preds_scaled_raw.

        Args:
            preds_scaled_raw (list or np.ndarray): List or array of raw predicted points in world frame.
        """
        if preds_scaled_smooth is not None and len(preds_scaled_smooth) > 0:
            arr = np.array(preds_scaled_smooth)  # Convert to numpy for plotting
            self.line_smooth.set_data(arr[:, 0], arr[:, 1])
        
        # Clear to empty list
        else:
            self.line_smooth.set_data([], [])
        self.fig.canvas.draw_idle()

    def update_sample_line(self):
        """
        Update the sampled trajectory line based on current sampled points.
        """
        if self.sampled is not None and len(self.sampled)>0:
            s=self.sampled
            self.line_samp.set_data(s[:,0], s[:,1])
        else:
            self.line_samp.set_data([],[])
        self.fig.canvas.draw_idle()

    def update_seed_line(self):
        """
        Update the seed trajectory line based on current sampled and seed_end.
        """
        if self.sampled is None or self.seed_end is None or self.seed_end < K_HIST-1:
            self.line_seed.set_data([],[])
        else:
            start_idx = self.seed_end - (K_HIST - 1)
            seg = self.sampled[start_idx : self.seed_end+1]
            self.line_seed.set_data(seg[:,0], seg[:,1])
        self.fig.canvas.draw_idle()

    def update_ref_pred_gt(self, preds=None, gt=None):
        """
        Update the reference prediction and ground truth lines.
        
        Args:
            preds (np.ndarray): Predicted trajectory points, shape (N, 2)
            gt (np.ndarray): Ground truth trajectory points, shape (N, 2)
        """
        if preds is not None and len(preds) > 0:
            self.line_pred.set_data(preds[:,0], preds[:,1])
        else:
            self.line_pred.set_data([], [])
        if gt is not None and len(gt) > 0:
            self.line_gt.set_data(gt[:,0], gt[:,1])
        else:
            self.line_gt.set_data([], [])
        self.fig.canvas.draw_idle()
