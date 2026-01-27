import torch
import numpy as np
import matplotlib.pyplot as plt

from config.runtime import (
    TRAIN_RATIO, SAMPLE_HZ, DEFAULT_SPEED, K_HIST, METHOD_ID
)
from geometry.frame3d import estimate_rotation_scale_3d, estimate_rotation_scale_3d_search_by_count
from geometry.resample import resample_trajectory_3d_equal_dt
from gp.dataset import build_dataset_3d, time_split
from gp.model import train_gp, rollout_reference_3d
from ui.handlers3d import on_press, on_move, on_release, on_key
from utils.misc import moving_average_centered, smooth_prediction_by_velocity


class DrawApp3D:
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

        self.ref_eq = None    # np.ndarray (N,3) resampled
        self.probe_eq = None  # np.ndarray (M,3) resampled

        # Model / alignment / prediction
        self.model_info = None
        self.R = None
        self.s = None
        self.t = None

        self.gt = None               # np.ndarray (H,3)
        self.preds = None            # np.ndarray (H,3)
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
        (self.line_ref_xy,) = self.ax_xy.plot([], [], lw=2.5, label="ref")
        (self.line_pred_xy,) = self.ax_xy.plot([], [], lw=2.0, label="pred")
        self.ax_xy.legend()

        (self.line_probe_yz,) = self.ax_yz.plot([], [], lw=2.0, label="probe")
        (self.line_pred_yz,) = self.ax_yz.plot([], [], lw=2.0, label="pred")
        self.ax_yz.legend()

        # Init lines (3D)
        (self.line_ref_3d,) = self.ax_3d.plot([], [], [], lw=3.0, label="ref")
        (self.line_probe_3d,) = self.ax_3d.plot([], [], [], lw=2.0, label="probe")
        (self.line_pred_3d,) = self.ax_3d.plot([], [], [], lw=2.0, label="pred")
        self.ax_3d.legend()

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
        self.line_ref_xy.set_color("tab:green")
        self.line_ref_xy.set_data(self.ref_eq[:, 0], self.ref_eq[:, 1])
        self.line_ref_3d.set_color("tab:green")
        self.line_ref_3d.set_data(self.ref_eq[:, 0], self.ref_eq[:, 1])
        self.line_ref_3d.set_3d_properties(self.ref_eq[:, 2])
        self.autoscale_3d()
        self.fig.canvas.draw_idle()

        print(f"[Train] Done. ref_eq={self.ref_eq.shape}, X={X.shape}, Y={Y.shape}")
        print()

    def process_probe_and_predict(
        self,
        *,
        k=K_HIST,
        input_type="spherical",
        output_type="delta",
        n_align=10,
        start_t=None,
    ):
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

        # R, s, t = estimate_rotation_scale_3d(self.ref_eq[:na], self.probe_eq[:na])
        R, s, t = estimate_rotation_scale_3d_search_by_count(
            self.ref_eq,
            self.probe_eq,
            s_min=0.5,
            s_max=2.0,
            margin_pts=200,
            step=1,
        )[:3]
        self.R, self.s, self.t = R, s, t
        # print(f"[Predict] Alignment done. R=\n{R}, s={s:.4f}, t={t}")

        # 3) Transform probe into ref frame
        probe_in_ref = ((self.probe_eq - t) / s) @ R
        self.probe_goal = self.s * (self.ref_eq[-1] @ self.R.T) + self.t

        # 4) Rollout in ref frame
        self.preds = []

        cur_hist = probe_in_ref.copy()
        rollout_horizon = self.rollout_horizon

        for step in range(rollout_horizon):
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
            self.preds.append(next_world)

            # Update every 5 steps
            if step % 5 == 0:
                self.update_pred_lines()

            # Update history
            cur_hist = np.vstack([cur_hist, next_ref])

            # Truncation Logic
            if self.probe_goal is not None:
                d = np.linalg.norm(next_world - self.probe_goal)
                if d < self.goal_stop_eps and np.max(vars_ref) > 1e-3:
                    print(f"Truncated at step {step}.")
                    break

        self.preds = np.asarray(self.preds, dtype=np.float64)

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

    def load_demo_spirals(self, *, T=400, turns=4*np.pi, radius=1.0, speed=0.1):
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
            [np.full_like(t, self.probe_plane_x) + speed * t,
            radius * np.cos(t),
            radius * np.sin(t)],
            axis=1
        )

        self.ref_raw = ref.tolist()
        self.probe_raw = probe[:50].tolist()

        # Reset derived buffers / outputs
        self.ref_eq = self.probe_eq = None
        self.model_info = None
        self.R = self.s = self.t = None
        self.preds = self.gt = None
        self.probe_goal = None

        self.update_ref_lines()
        self.update_probe_lines()
        self.update_pred_lines()

        print(f"[Demo] Loaded spirals: ref_raw={len(self.ref_raw)}, probe_raw={len(self.probe_raw)}")
        print()
    
    def update_ref_lines(self):
        """
        Update reference trajectory lines in all views.
        """
        if len(self.ref_raw) == 0:
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

        self.autoscale_3d()
        self.fig.canvas.draw_idle()

    def update_probe_lines(self):
        """
        Update probe trajectory lines in all views.
        """
        if len(self.probe_raw) == 0:
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

        self.autoscale_3d()
        self.fig.canvas.draw_idle()

    def update_pred_lines(self):
        """
        Update prediction trajectory lines in all views.
        """
        if self.preds is None or len(self.preds) == 0:
            self.line_pred_xy.set_data([], [])
            self.line_pred_yz.set_data([], [])
            self.line_pred_3d.set_data([], [])
            self.line_pred_3d.set_3d_properties([])
            self.autoscale_3d()
            self.fig.canvas.draw_idle()
            return

        P = np.asarray(self.preds, dtype=np.float64)

        # Show in XY and YZ views too (Projection)
        self.line_pred_xy.set_data(P[:, 0], P[:, 1])
        self.line_pred_yz.set_data(P[:, 1], P[:, 2])

        # 3D
        self.line_pred_3d.set_data(P[:, 0], P[:, 1])
        self.line_pred_3d.set_3d_properties(P[:, 2])

        self.autoscale_3d()
        self.fig.canvas.draw_idle()

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
        self.ref_raw = []
        self.probe_raw = []
        self.ref_eq = None
        self.probe_eq = None
        self.model_info = None
        self.R = self.s = self.t = None
        self.preds = None
        self.gt = None

        self.prediction_id += 1
        self.update_ref_lines()
        self.update_probe_lines()
        self.update_pred_lines()
        self.line_ref_xy.set_color("tab:blue")
        self.line_ref_3d.set_color("tab:blue")
        print("[UI] Cleared.")
        print()

    def show(self):
        plt.show()
