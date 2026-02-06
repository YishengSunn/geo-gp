import torch
import numpy as np
import matplotlib.pyplot as plt

from config.runtime import (
    TRAIN_RATIO, SAMPLE_HZ, DEFAULT_SPEED, K_HIST, METHOD_ID
)
from geometry.frame3d import estimate_rotation_scale_3d_search_by_count
from geometry.metrics import geom_mse
from geometry.resample import resample_trajectory_3d_equal_dt, resample_trajectory_6d_equal_dt
from gp.dataset import build_dataset_3d, build_dataset_6d, time_split
from gp.model import train_gp, rollout_reference_3d, rollout_reference_6d
from ui.handlers6d import on_press, on_move, on_release, on_key
from utils.misc import (
    moving_average_centered, moving_average_centered_6d, 
    smooth_prediction_by_velocity, smooth_prediction_by_twist_6d
)


class DrawApp6D:
    def __init__(self, *, probe_plane_x=2.0):
        """
        A simple UI for drawing 3D trajectories and predicting with GP.

        Args:
            probe_plane_x: float, x-coordinate of the probe drawing plane
            dt: float, time step for resampling
        """
        # Config
        self.probe_plane_x = float(probe_plane_x)  # X-coordinate of the probe drawing plane

        # Data buffers
        self.ref_raw = []     # list of xyz
        self.probe_raw = []   # list of xyz
        self.ref_rot_raw = None
        self.probe_rot_raw = None

        self.ref_eq = None    # np.ndarray (N,3) resampled
        self.probe_eq = None  # np.ndarray (M,3) resampled
        self.ref_rot_eq = None
        self.probe_rot_eq = None

        # Model / alignment / prediction
        self.model_info = None
        self.R = None
        self.s = None
        self.t = None

        self.gt = None               # np.ndarray (H,3)
        self.preds = None            # np.ndarray (H,3)
        self.preds_rot = None
        self.probe_goal = None       # np.ndarray (3,)
        self.goal_stop_eps = 0.05    # Stop if within this distance to goal
        self.rollout_horizon = 2000  # Max steps to rollout
        self.prediction_id = 0

        # Smoothing
        self.smooth_enabled = True
        self.smooth_win = 10

        # Drawing state
        self.drawing_ref = False
        self.drawing_probe = False
        self.use_6d = False

        # Build UI
        self.init_ui()

    def init_ui(self):
        self.fig = plt.figure(figsize=(13, 4))
        gs = self.fig.add_gridspec(1, 3)

        self.ax_xy = self.fig.add_subplot(gs[0, 0])
        self.ax_yz = self.fig.add_subplot(gs[0, 1])
        self.ax_3d = self.fig.add_subplot(gs[0, 2], projection="3d")

        self.ax_xy.set_title("Reference: draw on XY (z=0)  | Left mouse", fontsize=10)
        self.ax_yz.set_title(f"Probe: draw on YZ (x={self.probe_plane_x}) | Right mouse", fontsize=10)
        self.ax_3d.set_title("3D View")

        self.ax_xy.set_aspect("equal", adjustable="box")
        self.ax_yz.set_aspect("equal", adjustable="box")

        self.ax_xy.set_xlim(-1, 1)
        self.ax_xy.set_ylim(-1, 1)

        self.ax_yz.set_xlim(-1, 1)
        self.ax_yz.set_ylim(-1, 1)

        # Init lines (2D)
        (self.line_ref_xy,) = self.ax_xy.plot([], [], lw=2.5, c='r', label="ref")
        # Keep a handle to legend text for color sync
        self.ref_legend = None
        leg = self.ax_xy.legend()
        for legline, txt in zip(leg.legend_handles, leg.get_texts()):
            if txt.get_text() == "ref":
                self.ref_legend = legline
                break

        (self.line_probe_yz,) = self.ax_yz.plot([], [], lw=2.0, c='black', label="probe")
        (self.line_pred_yz,) = self.ax_yz.plot([], [], lw=2.0, c='tab:blue', label="pred")
        self.ax_yz.legend()

        # Init lines (3D)
        (self.line_ref_3d,) = self.ax_3d.plot([], [], [], lw=3.0, c='tab:red', label="ref")
        (self.line_probe_3d,) = self.ax_3d.plot([], [], [], lw=2.0, c='black', label="probe")
        (self.line_pred_3d,) = self.ax_3d.plot([], [], [], lw=2.0, c='tab:blue', label="pred")

        # Orientation quivers
        self.orient_quivers_ref = []
        self.orient_quivers_probe = []
        self.orient_quivers_pred = []

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", lambda event: on_press(self, event))
        self.fig.canvas.mpl_connect("motion_notify_event", lambda event: on_move(self, event))
        self.fig.canvas.mpl_connect("button_release_event", lambda event: on_release(self, event))
        self.fig.canvas.mpl_connect("key_press_event", lambda event: on_key(self, event))
    
    def train_reference(self, *, k=K_HIST, input_type="spherical", output_type="delta"):
        self.ref_eq = resample_trajectory_3d_equal_dt(self.ref_raw, sample_hz=SAMPLE_HZ, speed=DEFAULT_SPEED)

        if len(self.ref_eq) < k + 2:
            print(f"[Train] Need at least {k+2} resampled points, got {len(self.ref_eq)}!")
            print()
            return
        
        if self.smooth_enabled:
            self.ref_eq = moving_average_centered(self.ref_eq, self.smooth_win)
        ref_torch = torch.tensor(self.ref_eq, dtype=torch.float32)

        X, Y = build_dataset_3d(ref_torch, k, input_type=input_type, output_type=output_type)
        (Xtr, Ytr), (Xte, Yte), _ = time_split(X, Y, TRAIN_RATIO)

        self.model_info = train_gp(
            {"X_train": Xtr, "Y_train": Ytr, "X_test": Xte, "Y_test": Yte, "n_train": Xtr.shape[0]},
            METHOD_ID,
        )

        # Update reference lines to show resampled + smoothed
        self.ref_legend.set_color("tab:green")
        self.line_ref_xy.set_color("tab:green")
        self.line_ref_xy.set_data(self.ref_eq[:, 0], self.ref_eq[:, 1])
        self.line_ref_3d.set_color("tab:green")
        self.line_ref_3d.set_data(self.ref_eq[:, 0], self.ref_eq[:, 1])
        self.line_ref_3d.set_3d_properties(self.ref_eq[:, 2])
        self.autoscale_3d()
        self.fig.canvas.draw_idle()

        print(f"[Train] Done. ref_eq={self.ref_eq.shape}, X={X.shape}, Y={Y.shape}")
        print()

    def train_reference_6d(self, *, k=K_HIST, input_type="spherical", output_type="delta"):
        self.ref_eq, self.ref_rot_eq = resample_trajectory_6d_equal_dt(self.ref_raw, self.ref_rot_raw, 
                                                                       sample_hz=SAMPLE_HZ, speed=DEFAULT_SPEED)

        if len(self.ref_eq) < k + 2:
            print(f"[Train] Need at least {k+2} resampled points, got {len(self.ref_eq)}!")
            print()
            return
        
        if self.smooth_enabled:
            self.ref_eq = moving_average_centered_6d(self.ref_eq, self.smooth_win)
            self.ref_rot_eq = moving_average_centered_6d(self.ref_rot_eq, self.smooth_win)
        ref_torch = torch.tensor(self.ref_eq, dtype=torch.float32)
        ref_rot_torch = torch.tensor(self.ref_rot_eq, dtype=torch.float32)

        X, Y = build_dataset_6d(ref_torch, ref_rot_torch, k, input_type=input_type, output_type=output_type)
        (Xtr, Ytr), (Xte, Yte), _ = time_split(X, Y, TRAIN_RATIO)

        self.model_info = train_gp(
            {"X_train": Xtr, "Y_train": Ytr, "X_test": Xte, "Y_test": Yte, "n_train": Xtr.shape[0]},
            METHOD_ID,
        )

        # Update reference lines to show resampled + smoothed
        for q in self.orient_quivers_ref:
            q.remove()
        self.orient_quivers_ref.clear()
        self.draw_orientations(self.ref_eq, self.ref_rot_eq, self.orient_quivers_ref)

        self.ref_legend.set_color("tab:green")
        self.line_ref_xy.set_color("tab:green")
        self.line_ref_xy.set_data(self.ref_eq[:, 0], self.ref_eq[:, 1])
        self.line_ref_3d.set_color("tab:green")
        self.line_ref_3d.set_data(self.ref_eq[:, 0], self.ref_eq[:, 1])
        self.line_ref_3d.set_3d_properties(self.ref_eq[:, 2])
        self.autoscale_3d()
        self.fig.canvas.draw_idle()

        print(f"[Train] Done. ref_eq={self.ref_eq.shape}, ref_rot_eq={self.ref_rot_eq.shape}, X={X.shape}, Y={Y.shape}")
        print()

    def process_probe_and_predict(
        self,
        *,
        k=K_HIST,
        input_type="spherical",
        output_type="delta",
        n_align=10
    ):
        """
        Process the drawn probe trajectory and perform prediction.

        Args:
            k: int, history length for GP input
            input_type: str, type of input representation
            output_type: str, type of output representation
            n_align: int, number of points to use for alignment
        """
        # Cancel any running prediction
        self.prediction_id += 1
        local_pred_id = self.prediction_id

        if self.model_info is None or self.ref_eq is None:
            print("[Predict] Train first (press T)!")
            print()
            return

        # 1) Resample probe
        self.probe_eq = resample_trajectory_3d_equal_dt(self.probe_raw, sample_hz=SAMPLE_HZ, speed=DEFAULT_SPEED)
        print(f"[Predict] Resampled probe_eq: {self.probe_eq.shape}")
        if len(self.probe_eq) < (k + 2):
            print(f"[Predict] Not enough probe points. Need >= {k+2}, got {len(self.probe_eq)}!")
            print()
            return
        if self.smooth_enabled:
            self.probe_eq = moving_average_centered(self.probe_eq, self.smooth_win)

        # Update probe lines to show resampled + smoothed
        self.line_probe_yz.set_data(self.probe_eq[:, 1], self.probe_eq[:, 2])
        self.line_probe_3d.set_data(self.probe_eq[:, 0], self.probe_eq[:, 1])
        self.line_probe_3d.set_3d_properties(self.probe_eq[:, 2])
        self.fig.canvas.draw_idle()

        # 2) Alignment using first n_align points
        na = min(n_align, len(self.ref_eq), len(self.probe_eq))
        if na < 3:
            print("[Predict] Not enough points for alignment!")
            print()
            return

        R, s, t = estimate_rotation_scale_3d_search_by_count(
            self.ref_eq,
            self.probe_eq,
            s_min=0.5,
            s_max=2.0,
            margin_pts=300,
            step=10,
        )[:3]
        self.R, self.s, self.t = R, s, t
        # print(f"[Predict] Alignment done. R=\n{R}, s={s:.4f}, t={t}")

        # 3) Transform probe into ref frame
        probe_in_ref = ((self.probe_eq - t) / s) @ R
        self.probe_goal = self.s * (self.ref_eq[-1] @ self.R.T) + self.t

        # 4) Rollout in ref frame
        self.preds = None

        mse_thresh = 0.01
        drop_k = 5
        max_retries = 5

        for attempt in range(max_retries):
            cur_hist = probe_in_ref.copy()
            preds_world = []
            failed = False

            for step in range(self.rollout_horizon):
                if local_pred_id != self.prediction_id:
                    print("[Predict] Cancelled by new drawing.")
                    print()
                    return
                
                preds_ref, _, _, vars_ref = rollout_reference_3d(
                    self.model_info,
                    torch.tensor(cur_hist, dtype=torch.float32),
                    start_t=cur_hist.shape[0] - 1,
                    h=1,
                    k=k,
                    input_type=input_type,
                    output_type=output_type,
                )

                next_ref = preds_ref[-1].numpy()

                # Ref -> Probe/world
                next_world = self.s * (next_ref @ self.R.T) + self.t
                preds_world.append(next_world)

                # Update history
                cur_hist = np.vstack([cur_hist, next_ref])
 
                # Truncation Logic
                if self.probe_goal is not None:
                    d = np.linalg.norm(next_world - self.probe_goal)
                    if d < self.goal_stop_eps and np.max(vars_ref) > 1e-3:
                        print(f"[Predict] Reached goal at step {step}, d={d:.4f}")
                        break

            # Geometric drift check
            mse_full = geom_mse(cur_hist, self.ref_eq, min(len(cur_hist), len(self.ref_eq)))
            print(f"[GeomCheck] full mse = {mse_full:.4f}")

            if mse_full > mse_thresh:
                print("[Recover] Geometric drift detected, retry...")
                failed = True

            if not failed:
                preds_world = np.asarray(preds_world)

                # Robust start selection: enforce temporal consistency
                probe_end = self.probe_eq[-1]
                max_start_jump = 0.05

                dists = np.linalg.norm(preds_world - probe_end, axis=1)
                candidate_idxs = np.where(dists < max_start_jump)[0]
                
                if len(candidate_idxs) == 0:
                    print(f"[Recover] No prediction point close enough to probe end, retry...")
                    failed = True
                else:
                    # Choose the earliest such index to avoid skipping trajectory segments
                    i_start = int(candidate_idxs[0])
                    d_start = float(dists[i_start])
                    print(f"[Recover] Using earliest matching index {i_start}, jump={d_start:.4f}")

                    # Additional safety: ensure local continuity (no big velocity jump)
                    if i_start > 0:
                        step_jump = np.linalg.norm(preds_world[i_start] - preds_world[i_start - 1])
                        if step_jump > 3.0 * np.mean(np.linalg.norm(np.diff(preds_world, axis=0), axis=1)):
                            print("[Recover] Detected discontinuity before start point, retry...")
                            failed = True
                            continue

                    self.preds = preds_world[i_start:]
                    break

            # Drop tail of probe history and retry
            if probe_in_ref.shape[0] <= (k + drop_k):
                break

            probe_in_ref = probe_in_ref[:-drop_k]
            print(f"[Recover] Dropping last {drop_k} probe points, retry {attempt+1}")

        if self.preds is None:
            print("[Predict] All retries failed. No prediction output.")
            print()
            return

        # 5) Smooth prediction by velocity
        if self.smooth_enabled:
            self.preds = smooth_prediction_by_velocity(
                probe=self.probe_eq,
                pred=self.preds,
                win=self.smooth_win,
                blend_first_step=0.5,
            )

        self.update_pred_lines()
        print(f"[Predict] Done. preds={self.preds.shape}")
        print()

    def process_probe_and_predict_6d(
        self,
        *,
        k=K_HIST,
        input_type="spherical",
        output_type="delta",
        n_align=10
    ):
        """
        Process the drawn probe trajectory with orientations and perform 6D prediction.

        Args:
            k: int, history length for GP input
            input_type: str, type of input representation
            output_type: str, type of output representation
            n_align: int, number of points to use for alignment
        """
        # Cancel any running prediction
        self.prediction_id += 1
        local_pred_id = self.prediction_id

        if self.model_info is None or self.ref_eq is None or self.ref_rot_eq is None:
            print("[Predict] Train first!")
            print()
            return

        # 1) Resample probe (position + orientation)
        self.probe_eq, self.probe_rot_eq = resample_trajectory_6d_equal_dt(
            self.probe_raw,
            self.probe_rot_raw,
            sample_hz=SAMPLE_HZ,
            speed=DEFAULT_SPEED,
        )

        if len(self.probe_eq) < (k + 2):
            print(f"[Predict] Not enough probe points. Need >= {k+2}, got {len(self.probe_eq)}!")
            print()
            return

        if self.smooth_enabled:
            self.probe_eq = moving_average_centered_6d(self.probe_eq, self.smooth_win)
            self.probe_rot_eq = moving_average_centered_6d(self.probe_rot_eq, self.smooth_win)

        # Update probe lines
        for q in self.orient_quivers_probe:
            q.remove()
        self.orient_quivers_probe.clear()
        self.draw_orientations(self.probe_eq, self.probe_rot_eq, self.orient_quivers_probe)

        self.line_probe_yz.set_data(self.probe_eq[:, 1], self.probe_eq[:, 2])
        self.line_probe_3d.set_data(self.probe_eq[:, 0], self.probe_eq[:, 1])
        self.line_probe_3d.set_3d_properties(self.probe_eq[:, 2])
        self.fig.canvas.draw_idle()

        # 2) Alignment
        R, s, t = estimate_rotation_scale_3d_search_by_count(
            self.ref_eq,
            self.probe_eq,
            s_min=0.5,
            s_max=2.0,
            margin_pts=300,
            step=10,
        )[:3]
        self.R, self.s, self.t = R, s, t

        # 3) Transform probe into ref frame
        probe_in_ref = ((self.probe_eq - t) / s) @ R
        self.probe_goal = self.s * (self.ref_eq[-1] @ self.R.T) + self.t
        probe_rot_in_ref = R.T @ self.probe_rot_eq

        # 4) Rollout in ref frame
        mse_thresh = 0.01
        drop_k = 5
        max_retries = 5

        self.preds = None
        self.preds_rot = None

        for attempt in range(max_retries):
            cur_pos = probe_in_ref.copy()
            cur_rot = probe_rot_in_ref.copy()

            preds_world_pos = []
            preds_world_rot = []
            failed = False

            for step in range(self.rollout_horizon):
                if local_pred_id != self.prediction_id:
                    print("[Predict] Cancelled by new drawing.")
                    print()
                    return
                
                preds_ref_pos, preds_ref_rot, _, _, vars_ref = rollout_reference_6d(
                    self.model_info,
                    torch.tensor(cur_pos, dtype=torch.float32),
                    torch.tensor(cur_rot, dtype=torch.float32),
                    start_t=cur_pos.shape[0] - 1,
                    h=1,
                    k=k,
                    input_type=input_type,
                    output_type=output_type,
                )

                next_ref_pos = preds_ref_pos[-1].numpy()
                next_ref_rot = preds_ref_rot[-1].numpy()

                next_world_pos = self.s * (next_ref_pos @ self.R.T) + self.t
                next_world_rot = self.R @ next_ref_rot

                preds_world_pos.append(next_world_pos)
                preds_world_rot.append(next_world_rot)

                # Update history in ref frame
                cur_pos = np.vstack([cur_pos, next_ref_pos])
                cur_rot = np.vstack([cur_rot, next_ref_rot[None, :, :]])

                # Truncation logic
                if self.probe_goal is not None:
                    d = np.linalg.norm(next_world_pos - self.probe_goal)
                    if d < self.goal_stop_eps and np.max(vars_ref) > 1e-3:
                        print(f"[Predict] Reached goal at step {step}, d={d:.4f}")
                        break

            # Geometric drift check (position only)
            mse_full = geom_mse(cur_pos, self.ref_eq, min(len(cur_pos), len(self.ref_eq)))
            print(f"[GeomCheck] full mse = {mse_full:.4f}")

            if mse_full > mse_thresh:
                print("[Recover] Geometric drift detected, retry...")
                failed = True

            if not failed:
                preds_world_pos = np.asarray(preds_world_pos)
                preds_world_rot = np.asarray(preds_world_rot)

                # Robust start selection near probe end
                probe_end = self.probe_eq[-1]
                max_start_jump = 0.05

                dists = np.linalg.norm(preds_world_pos - probe_end, axis=1)
                candidate_idxs = np.where(dists < max_start_jump)[0]

                if len(candidate_idxs) == 0:
                    print("[Recover] No prediction point close to probe end, retry...")
                    failed = True
                else:
                    i_start = int(candidate_idxs[0])
                    print(f"[Recover] Using earliest matching index {i_start}")

                    # Continuity check
                    if i_start > 0:
                        step_jump = np.linalg.norm(preds_world_pos[i_start] - preds_world_pos[i_start - 1])
                        mean_step = np.mean(np.linalg.norm(np.diff(preds_world_pos, axis=0), axis=1))
                        if step_jump > 3.0 * mean_step:
                            print("[Recover] Discontinuity detected, retry...")
                            failed = True
                            continue

                    self.preds = preds_world_pos[i_start:]
                    self.preds_rot = preds_world_rot[i_start:]
                    break

            # Drop tail and retry
            if probe_in_ref.shape[0] <= (k + drop_k):
                break

            probe_in_ref = probe_in_ref[:-drop_k]
            probe_rot_in_ref = probe_rot_in_ref[:-drop_k]
            print(f"[Recover] Dropping last {drop_k} probe points, retry {attempt+1}")

        if self.preds is None:
            print("[Predict] All retries failed. No prediction output.")
            return

        # 5) Smooth prediction by twist
        if self.smooth_enabled:
            self.preds, self.preds_rot = smooth_prediction_by_twist_6d(
                probe_pos=self.probe_eq,
                probe_rot=self.probe_rot_eq,
                pred_pos=self.preds,
                pred_rot=self.preds_rot,
                win=self.smooth_win,
            )

        self.update_pred_lines()
        print(f"[Predict] Done. preds={self.preds.shape}")
        print()

    def update_ref_lines(self):
        """
        Update reference trajectory lines in all views.
        """
        if len(self.ref_raw) == 0:
            for q in self.orient_quivers_ref:
                q.remove()
            self.orient_quivers_ref.clear()

            self.line_ref_xy.set_data([], [])
            self.line_ref_3d.set_data([], [])
            self.line_ref_3d.set_3d_properties([])
            self.autoscale_3d()
            self.fig.canvas.draw_idle()
            return

        P = np.asarray(self.ref_raw, dtype=np.float64)
        # XY view
        self.line_ref_xy.set_data(P[:, 0], P[:, 1])

        # 3D view
        self.line_ref_3d.set_data(P[:, 0], P[:, 1])
        self.line_ref_3d.set_3d_properties(P[:, 2])

        if self.ref_rot_raw is not None and len(P) == len(self.ref_rot_raw):
            self.draw_orientations(P, self.ref_rot_raw, self.orient_quivers_ref)
        else:
            for q in self.orient_quivers_ref:
                q.remove()
            self.orient_quivers_ref.clear()

        self.autoscale_3d()
        self.fig.canvas.draw_idle()

    def update_probe_lines(self):
        """
        Update probe trajectory lines in all views.
        """
        if len(self.probe_raw) == 0:
            for q in self.orient_quivers_probe:
                q.remove()
            self.orient_quivers_probe.clear()

            self.line_probe_yz.set_data([], [])
            self.line_probe_3d.set_data([], [])
            self.line_probe_3d.set_3d_properties([])
            self.autoscale_3d()
            self.fig.canvas.draw_idle()
            return

        P = np.asarray(self.probe_raw, dtype=np.float64)
        # YZ view (x axis = y, y axis = z)
        self.line_probe_yz.set_data(P[:, 1], P[:, 2])

        # 3D view
        self.line_probe_3d.set_data(P[:, 0], P[:, 1])
        self.line_probe_3d.set_3d_properties(P[:, 2])

        if self.probe_rot_raw is not None and len(P) == len(self.probe_rot_raw):
            self.draw_orientations(P, self.probe_rot_raw, self.orient_quivers_probe)
        else:
            for q in self.orient_quivers_probe:
                q.remove()
            self.orient_quivers_probe.clear()

        self.autoscale_3d()
        self.fig.canvas.draw_idle()

    def update_pred_lines(self):
        """
        Update prediction trajectory lines in all views.
        """
        if self.preds is None or len(self.preds) == 0:
            self.line_pred_yz.set_data([], [])
            self.line_pred_3d.set_data([], [])
            self.line_pred_3d.set_3d_properties([])
            self.autoscale_3d()
            self.fig.canvas.draw_idle()
            return

        P = np.asarray(self.preds, dtype=np.float64)

        # Show in XY and YZ views too (Projection)
        self.line_pred_yz.set_data(P[:, 1], P[:, 2])

        # 3D
        self.line_pred_3d.set_data(P[:, 0], P[:, 1])
        self.line_pred_3d.set_3d_properties(P[:, 2])

        if self.preds_rot is not None and len(self.preds_rot) == len(self.preds):
            self.draw_orientations(self.preds, self.preds_rot, self.orient_quivers_pred)
        else:
            for q in self.orient_quivers_pred:
                q.remove()
            self.orient_quivers_pred.clear()

        self.autoscale_3d()
        self.fig.canvas.draw_idle()

    def draw_orientations(self, positions, rotations, quiver_store, scale=0.1):
        """
        Draw orientation arrows in 3D using the local X, Y, Z axes of each rotation matrix.

        Args:
            positions: list or np.ndarray of shape (N,3), positions of the quivers
            rotations: list or np.ndarray of shape (N,3,3), rotation matrices for the quivers
            quiver_store: list, to store the created quiver objects for later removal
            scale: float, scaling factor for the quiver lengths
        """
        # Clear previous
        for q in quiver_store:
            q.remove()
        quiver_store.clear()

        if rotations is None:
            return

        P = np.asarray(positions, dtype=np.float64)
        R = np.asarray(rotations, dtype=np.float64)

        for p, Rm in zip(P[::10], R[::10]):  # Draw sparsely
            # Local axes from rotation matrix
            x_dir = Rm[:, 0] * scale
            y_dir = Rm[:, 1] * scale
            z_dir = Rm[:, 2] * scale

            # Draw X axis (red)
            qx = self.ax_3d.quiver(
                p[0], p[1], p[2],
                x_dir[0], x_dir[1], x_dir[2],
                length=1.0, normalize=False, color='r'
            )
            quiver_store.append(qx)

            # Draw Y axis (green)
            qy = self.ax_3d.quiver(
                p[0], p[1], p[2],
                y_dir[0], y_dir[1], y_dir[2],
                length=1.0, normalize=False, color='g'
            )
            quiver_store.append(qy)

            # Draw Z axis (blue)
            qz = self.ax_3d.quiver(
                p[0], p[1], p[2],
                z_dir[0], z_dir[1], z_dir[2],
                length=1.0, normalize=False, color='b'
            )
            quiver_store.append(qz)

    def autoscale_3d(self, margin=0.05):
        """
        Keep 3D view in true equal scale based on all currently available points.

        Args:
            margin: float, margin ratio to add around the points
        """
        pts_list = []

        if len(self.ref_raw) > 0:
            pts_list.append(np.asarray(self.ref_raw, dtype=np.float64))
        if len(self.probe_raw) > 0:
            pts_list.append(np.asarray(self.probe_raw, dtype=np.float64))
        if self.preds is not None and len(self.preds) > 0:
            pts_list.append(np.asarray(self.preds, dtype=np.float64))

        if len(pts_list) == 0:
            return

        P = np.vstack(pts_list)

        xmin, ymin, zmin = P.min(axis=0)
        xmax, ymax, zmax = P.max(axis=0)

        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin

        max_range = max(dx, dy, dz)
        eps = 1e-6
        if max_range < eps:
            max_range = 1.0

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)

        r = 0.5 * max_range * (1.0 + margin)

        self.ax_3d.set_xlim(cx - r, cx + r)
        self.ax_3d.set_ylim(cy - r, cy + r)
        self.ax_3d.set_zlim(cz - r, cz + r)

    def clear(self):
        self.ref_raw = self.probe_raw = []
        self.ref_rot_raw = self.probe_rot_raw = None
        self.ref_eq = self.probe_eq = None
        self.ref_rot_eq = self.probe_rot_eq = None
        self.preds = self.gt = None
        self.model_info = None
        self.probe_goal = None
        self.R = self.s = self.t = None
        self.prediction_id += 1

        # Remove existing orientation quivers from the plot
        for q in self.orient_quivers_ref:
            try:
                q.remove()
            except Exception:
                pass
        self.orient_quivers_ref.clear()
        for q in self.orient_quivers_probe:
            try:
                q.remove()
            except Exception:
                pass
        self.orient_quivers_probe.clear()
        for q in self.orient_quivers_pred:
            try:
                q.remove()
            except Exception:
                pass
        self.orient_quivers_pred.clear()

        self.update_ref_lines()
        self.update_probe_lines()
        self.update_pred_lines()
        self.ref_legend.set_color("tab:red")
        self.line_ref_xy.set_color("tab:red")
        self.line_ref_3d.set_color("tab:red")
        print("[UI] Cleared.")
        print()

    def show(self):
        plt.show()
