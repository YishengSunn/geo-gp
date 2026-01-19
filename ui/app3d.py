import torch
import numpy as np
import matplotlib.pyplot as plt

from config.runtime import TRAIN_RATIO, K_HIST, METHOD_ID
from geometry.frame3d import estimate_rotation_scale_3d
from gp.dataset import build_dataset_3d, time_split
from gp.model import train_gp, rollout_reference_3d


class App3D:
    def __init__(self):
        # Trajectories
        self.ref_traj = None      # torch.Tensor (T,3)
        self.probe_traj = None    # torch.Tensor (T,3)

        # GP model
        self.model_info = None

        # Alignment
        self.R = None
        self.s = None
        self.t = None

        # Prediction
        self.preds = None
        self.gt = None

        # Visualization
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_box_aspect([1, 1, 1])

        self.line_ref, = self.ax.plot([], [], [], lw=3, color="g", label="Reference")
        self.line_probe, = self.ax.plot([], [], [], lw=1.5, color="k", label="Probe")
        self.line_pred, = self.ax.plot([], [], [], lw=1.5, color="b", label="Prediction")

        self.ax.legend()

    def set_reference(self, traj: torch.Tensor):
        """
        Set reference trajectory.

        Args:
            traj: torch.Tensor of shape (T, 3)
        """
        assert traj.ndim == 2 and traj.shape[1] == 3
        self.ref_traj = traj.clone()

    def train_reference_gp(self, k=K_HIST, input_type="spherical", output_type="delta"):
        """
        Train GP model on reference trajectory.
        
        Args:
            k: int, history length
            input_type: str, input data type
            output_type: str, output data type
        """
        assert self.ref_traj is not None, "Reference trajectory not set!"

        X, Y = build_dataset_3d(self.ref_traj, k, input_type=input_type, output_type=output_type)

        (Xtr, Ytr), (Xte, Yte), _ = time_split(X, Y, TRAIN_RATIO)

        self.model_info = train_gp(
            {
                "X_train": Xtr,
                "Y_train": Ytr,
                "X_test": Xte,
                "Y_test": Yte,
                "n_train": Xtr.shape[0],
            },
            METHOD_ID,
        )

    def set_probe(self, traj: torch.Tensor):
        """
        Set probe trajectory.

        Args:
            traj: torch.Tensor of shape (T, 3)
        """
        assert traj.ndim == 2 and traj.shape[1] == 3
        self.probe_traj = traj.clone()

    def estimate_alignment(self, n_align=20):
        """
        Estimate alignment (R, s, t) from probe to reference using first n_align points.

        Args:
            n_align: int, number of points to use for alignment
        """
        assert self.ref_traj is not None
        assert self.probe_traj is not None

        ref_np = self.ref_traj[:n_align].cpu().numpy()
        probe_np = self.probe_traj[:n_align].cpu().numpy()

        self.R, self.s, self.t = estimate_rotation_scale_3d(ref_np, probe_np)

    def rollout(
        self,
        start_t,
        horizon,
        input_type="pos",
        output_type="delta",
    ):
        """
        Rollout GP model from probe trajectory.

        Args:
            start_t: int, starting time index for rollout
            horizon: int, number of steps to rollout
            input_type: str, input data type
            output_type: str, output data type
        """
        assert self.model_info is not None
        assert self.R is not None, "Alignment not estimated!"

        # probe -> reference frame
        probe_np = self.probe_traj.cpu().numpy()
        probe_in_ref = ((probe_np - self.t) / self.s) @ self.R

        probe_in_ref = torch.tensor(probe_in_ref, dtype=torch.float32)

        preds_ref, gt_ref, _, _ = rollout_reference_3d(
            self.model_info,
            probe_in_ref,
            start_t,
            horizon,
            K_HIST,
            input_type=input_type,
            output_type=output_type,
        )

        # reference -> probe/world frame
        preds_world = self.s * (preds_ref.numpy() @ self.R.T) + self.t
        gt_world = self.s * (gt_ref.numpy() @ self.R.T) + self.t

        self.preds = torch.tensor(preds_world, dtype=torch.float32)
        self.gt = torch.tensor(gt_world, dtype=torch.float32)

    def autoscale_3d(self, pts: np.ndarray, margin=0.05):
        """
        Set equal 3D axis limits based on points.

        Args:
            pts: np.ndarray of shape (N, 3)
            margin: float, margin ratio to add around the points
        """
        xmin, ymin, zmin = pts.min(axis=0)
        xmax, ymax, zmax = pts.max(axis=0)

        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin

        max_range = max(dx, dy, dz)
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)

        r = 0.5 * max_range * (1.0 + margin)

        self.ax.set_xlim(cx - r, cx + r)
        self.ax.set_ylim(cy - r, cy + r)
        self.ax.set_zlim(cz - r, cz + r)

    def plot(self, start_t):
        """
        Plot reference, probe, and prediction trajectories.

        Args:
            start_t: int, starting time index for probe trajectory
        """
        # Reference
        self.line_ref.set_data(self.ref_traj[:, 0], self.ref_traj[:, 1])
        self.line_ref.set_3d_properties(self.ref_traj[:, 2])

        # Probe
        self.line_probe.set_data(self.probe_traj[:start_t+1, 0], self.probe_traj[:start_t+1, 1])
        self.line_probe.set_3d_properties(self.probe_traj[:start_t+1, 2])

        # Prediction
        if self.preds is not None:
            self.line_pred.set_data(self.preds[:, 0], self.preds[:, 1])
        self.line_pred.set_3d_properties(self.preds[:, 2])

        self.autoscale_3d(np.vstack([self.ref_traj.numpy(), self.probe_traj.numpy(), self.preds.numpy()]))
        self.fig.canvas.draw_idle()
        plt.show()
