import os
import torch
import numpy as np

from config.runtime import SAMPLE_HZ, DEFAULT_SPEED, METHOD_ID
from geometry.resample import resample_trajectory_3d_equal_dt, resample_trajectory_6d_equal_dt
from gp.dataset import build_dataset_3d, build_dataset_6d, time_split
from gp.model import train_gp
from utils.misc import moving_average_centered_pos, moving_average_centered_6d


class Skill:
    """
    A single manipulation skill.

    Each skill contains:
        - reference trajectory
        - GP model trained from the trajectory
    """

    def __init__(
        self,
        name: str,
        ref_pos: np.ndarray,
        ref_quat: np.ndarray,
        *,
        ref_force: np.ndarray | None = None,
        mode: str = "6d",
        smooth: bool = True,
        smooth_win: int = 15,
    ):

        """
        Initialize a skill.

        Args:
            name: str, name of the skill
            ref_pos: np.ndarray, (N,3) reference positions
            ref_quat: np.ndarray, (N,4) reference quaternions
            ref_force: np.ndarray | None, optional (N,3) force samples aligned with ref_pos (same frame)
            mode: str, mode of the skill ("3d" or "6d")
            smooth: bool, whether to apply smoothing to the reference trajectory
            smooth_win: int, window size for smoothing (if enabled)
        """
        self.name = name
        self.mode = mode

        # Raw demonstrations
        self.ref_raw = np.asarray(ref_pos, dtype=np.float64)
        self.ref_quat_raw = np.asarray(ref_quat, dtype=np.float64)
        self.ref_force_raw = None if ref_force is None else np.asarray(ref_force, dtype=np.float64)

        # Resampled trajectories
        self.ref_eq = None
        self.ref_quat_eq = None
        self.ref_force_eq = None

        # GP model
        self.model = None

        # Config
        self.smooth = smooth
        self.smooth_win = smooth_win

    # Prepare reference trajectory
    def prepare_reference(self):
        """
        Resample the reference trajectory to have equal time intervals, and optionally apply smoothing.
        """

        if self.mode == "6d":
            if self.ref_force_raw is not None:
                self.ref_eq, self.ref_quat_eq, self.ref_force_eq = resample_trajectory_6d_equal_dt(
                    self.ref_raw,
                    self.ref_quat_raw,
                    sample_hz=SAMPLE_HZ,
                    speed=DEFAULT_SPEED,
                    points_force=self.ref_force_raw,
                )
            else:
                self.ref_eq, self.ref_quat_eq = resample_trajectory_6d_equal_dt(
                    self.ref_raw,
                    self.ref_quat_raw,
                    sample_hz=SAMPLE_HZ,
                    speed=DEFAULT_SPEED,
                )
                self.ref_force_eq = None

            if self.smooth:
                self.ref_eq = moving_average_centered_6d(self.ref_eq, self.smooth_win)
                self.ref_quat_eq = moving_average_centered_6d(self.ref_quat_eq, self.smooth_win)
                if self.ref_force_eq is not None:
                    self.ref_force_eq = moving_average_centered_6d(self.ref_force_eq, self.smooth_win)

        elif self.mode == "3d":
            self.ref_eq = resample_trajectory_3d_equal_dt(
                self.ref_raw,
                sample_hz=SAMPLE_HZ,
                speed=DEFAULT_SPEED,
            )

            if self.smooth:
                self.ref_eq = moving_average_centered_pos(self.ref_eq, self.smooth_win)

    # Train GP model
    def train_gp(
        self,
        *,
        k: int,
        input_type: str = "spherical",
        output_type: str = "delta",
        train_ratio: float = 1.0,
    ):
        """
        Train a GP model for this skill using the reference trajectory.

        Args:
            k: int, number of past time steps to use as input
            input_type: str, how to represent the input (e.g., "spherical", "cartesian")
            output_type: str, how to represent the output (e.g., "delta", "absolute")
            train_ratio: float, ratio of data to use for training vs testing
        """
        if self.ref_eq is None:
            self.prepare_reference()

        if self.mode == "6d":
            ref_pos = torch.tensor(self.ref_eq, dtype=torch.float32)
            ref_quat = torch.tensor(self.ref_quat_eq, dtype=torch.float32)
            ref_force = None if self.ref_force_eq is None else torch.tensor(self.ref_force_eq, dtype=torch.float32)

            X, Y = build_dataset_6d(
                ref_pos,
                ref_quat,
                k,
                input_type=input_type,
                output_type=output_type,
                traj_force=ref_force,
            )

        elif self.mode == "3d":
            ref_pos = torch.tensor(self.ref_eq, dtype=torch.float32)
            
            X, Y = build_dataset_3d(
                ref_pos,
                k,
                input_type=input_type,
                output_type=output_type,
            )

        (Xtr, Ytr), (Xte, Yte), _ = time_split(X, Y, train_ratio)

        self.model = train_gp(
            {
                "X_train": Xtr,
                "Y_train": Ytr,
                "X_test": Xte,
                "Y_test": Yte,
                "n_train": Xtr.shape[0],
            },
            METHOD_ID,
        )

        print(f"[Skill] Trained GP for skill: {self.name}, points: {len(self)}")
        print()

    def save(self, folder: str):
        """
        Save the skill to a file.

        Args:
            folder: str, directory to save the skill file
        """
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.name}.pt")

        payload = {
            "ref_eq": self.ref_eq,
            "ref_quat_eq": self.ref_quat_eq,
            "model": self.model,
        }
        if self.ref_force_eq is not None:
            payload["ref_force_eq"] = self.ref_force_eq
        torch.save(payload, path)

    def load(self, path: str):
        """
        Load the skill from a file.

        Args:
            path: str, path to the skill file
        """
        data = torch.load(path, weights_only=False)

        self.ref_eq = data["ref_eq"]
        self.ref_quat_eq = data["ref_quat_eq"]
        self.ref_force_eq = data.get("ref_force_eq", None)
        self.model = data["model"]

    # Info
    def __len__(self) -> int:
        if self.ref_eq is None:
            return 0
        return len(self.ref_eq)

    def __repr__(self) -> str:
        n = len(self)

        return (
            f"Skill(name={self.name}, "
            f"points={n}, "
            f"trained={self.model is not None})"
        )
