import os
import numpy as np
import pandas as pd

from skills.skill import Skill


# Load one skill from CSV
def load_skill_csv(path: str, mode: str = "6d") -> Skill:
    """
    Load a skill from a CSV file. The CSV is expected to have columns: x,y,z,qx,qy,qz,qw 
    where (x,y,z) are positions and (qx,qy,qz,qw) are quaternions (in [x,y,z,w] format).

    Args:
        path: str, path to the CSV file
        mode: str, mode of the skill ("3d" or "6d")

    Returns:
        skill: skill object initialized with the data from the CSV
    """
    df = pd.read_csv(path)

    pos = df[["x", "y", "z"]].to_numpy(dtype=np.float64)

    qx = df["qx"].to_numpy(dtype=np.float64)
    qy = df["qy"].to_numpy(dtype=np.float64)
    qz = df["qz"].to_numpy(dtype=np.float64)
    qw = df["qw"].to_numpy(dtype=np.float64)

    quat = np.stack([qw, qx, qy, qz], axis=1)

    name = os.path.splitext(os.path.basename(path))[0]

    ref_force = None
    if all(c in df.columns for c in ("fx", "fy", "fz")):
        ref_force = df[["fx", "fy", "fz"]].to_numpy(dtype=np.float64)

    skill = Skill(
        name=name,
        ref_pos=pos,
        ref_quat=quat,
        ref_force=ref_force,
        mode=mode,
    )

    return skill

# Load all CSV skills
def load_skills_from_folder(folder: str, mode: str = "6d") -> list[Skill]:
    """
    Load all skill CSV files from a folder. Each CSV file should represent one skill.

    Args:
        folder: str, path to the folder containing skill CSV files

    Returns:
        skills: list of skill objects loaded from the CSV files
    """
    skills = []

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".csv"):
            continue

        path = os.path.join(folder, fname)

        skill = load_skill_csv(path, mode=mode)

        skills.append(skill)

    print(f"[SkillLoader] Loaded {len(skills)} skills")

    return skills

def load_skills_from_models(folder: str, mode: str = "3d") -> list[Skill]:
    """
    Load all pretrained skill models from a folder. Each .pt file should represent one skill.

    Args:
        folder: str, path to the folder containing pretrained skill .pt files
        mode: str, mode of the skills ("3d" or "6d")

    Returns:
        skills: list of skill objects loaded from the pretrained models
    """
    skills = []

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".pt"):
            continue

        path = os.path.join(folder, fname)
        name = os.path.splitext(fname)[0]

        skill = Skill(name=name, ref_pos=[], ref_quat=[], mode=mode)
        skill.load(path)

        skills.append(skill)

    print(f"[SkillLoader] Loaded {len(skills)} pretrained skills")

    return skills

# Load + train skills
def load_and_train_skills(
    folder: str,
    *,
    k: int,
    mode: str = "6d",
    input_type: str = "spherical",
    output_type: str = "delta",
):
    """
    Load all skills from a folder and train GP models for each skill.

    Args:
        folder: str, path to the folder containing skill CSV files
        k: int, number of past time steps to use as input for GP training
        input_type: str, how to represent the input for GP training
        output_type: str, how to represent the output for GP training
    
    Returns:
        skills: list of skill objects with trained GP models
    """
    skills = load_skills_from_folder(folder, mode=mode)

    for skill in skills:
        print(f"[SkillLoader] Training skill: {skill.name}")

        skill.train_gp(
            k=k,
            input_type=input_type,
            output_type=output_type,
        )

    return skills
