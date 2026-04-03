import numpy as np

from geometry.frame6d import estimate_rotation_scale_3d_search_by_count
from skills.skill import Skill


class SkillLibrary:
    """
    A library of manipulation skills.

    The library can:
        - store multiple skills
        - train all skills
        - match a probe trajectory to the best skill
    """

    def __init__(self):
        self.skills = []

    # Add skill
    def add_skill(self, skill: Skill):
        """
        Add a skill to the library.
        
        Args:
            skill: skill object to add
        """
        self.skills.append(skill)

    # Train all skills
    def train_all(
        self,
        *,
        k: int,
        input_type: str = "spherical",
        output_type: str = "delta",
    ):
        """
        Train GP models for all skills in the library.
        
        Args:
            k: int, number of past time steps to use as input for GP training
            input_type: str, how to represent the input for GP training
            output_type: str, how to represent the output for GP training
        """
        for skill in self.skills:
            print(f"[SkillLibrary] Training {skill.name}")

            skill.train_gp(
                k=k,
                input_type=input_type,
                output_type=output_type,
            )

    # Matching
    def match(
        self,
        probe_eq: np.ndarray,
        *,
        margin_pts: int = 300,
        step: int = 10,
    ) -> tuple[Skill, tuple[np.ndarray, float, float, float, int]]:
        """
        Match a probe trajectory to the best skill in the library.

        Args:
            probe_eq: (N,7) probe trajectory in 6D (pos+quat)
            margin_pts: int, number of points to use for matching (from the start of the trajectory)
            step: int, step size for searching over alignment parameters

        Returns:
            best_skill: skill object that best matches the probe trajectory
            best_align: alignment parameters (R, s, t, j_end) for the best skill
        """
        if len(self.skills) == 0:
            raise RuntimeError("SkillLibrary is empty")

        best_skill = None
        best_align = None
        best_mse = np.inf

        for skill in self.skills:
            ref_eq = skill.ref_eq

            # Alignment
            R, s, t, j_end, rmse = estimate_rotation_scale_3d_search_by_count(
                ref_eq,
                probe_eq,
                margin_pts=margin_pts,
                step=step,
            )[:5]

            print(f"[SkillLibrary] {skill.name} rmse = {rmse:.6f}")

            # Update best
            if rmse < best_mse:
                best_mse = rmse
                best_skill = skill
                best_align = (R, s, t, j_end)

        print(f"[SkillLibrary] Selected skill: {best_skill.name}")
        print()

        return best_skill, best_align

    # Info
    def __len__(self) -> int:
        return len(self.skills)

    def __repr__(self) -> str:
        return f"SkillLibrary(n_skills={len(self.skills)})"
