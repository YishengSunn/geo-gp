"""Plot aligned reference, prompt, and prediction trajectory components."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config.runtime import DEFAULT_SPEED, SAMPLE_HZ
from geometry.resample import resample_trajectory_6d_equal_dt


COMPONENTS = ("x", "y", "z", "qx", "qy", "qz", "qw")
REF_PREFIXES = ("matched_ref", "matched_reference", "reference", "ref")
PROMPT_PREFIX = "prompt"
PRED_PREFIX = "prediction"
TRANSFORM_PREFIX = "similarity_transform"
PRED_EXTRA_POINTS_AFTER_REF = 2000
PLOT_RESAMPLE_SPEED_SCALE = 0.5


def load_pose_csv(path: Path) -> dict[str, np.ndarray]:
    """Load pose components from a CSV file.

    Args:
        path: CSV file containing pose component columns.

    Returns:
        Mapping from component name to a float array.

    Raises:
        ValueError: If any required pose columns are missing.
    """
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = COMPONENTS
    missing = [name for name in required if name not in reader.fieldnames]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in required
    }


def load_similarity_transform(path: Path) -> tuple[np.ndarray, float, np.ndarray, dict]:
    """Load a similarity transform from JSON.

    Args:
        path: JSON file containing ``R``, ``s``, and ``t`` fields.

    Returns:
        Rotation matrix, scale, translation vector, and raw JSON data.

    Raises:
        ValueError: If transform shapes are invalid or scale is near zero.
    """
    with path.open() as f:
        data = json.load(f)

    R = np.asarray(data["R"], dtype=np.float64)
    s = float(data["s"])
    t = np.asarray(data["t"], dtype=np.float64)

    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError(f"Invalid transform shapes in {path}: R={R.shape}, t={t.shape}")
    if abs(s) < 1e-12:
        raise ValueError(f"Invalid near-zero scale in {path}: {s}")

    return R, s, t, data


def poses_to_arrays(pose: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Convert a pose mapping to position and quaternion arrays.

    Args:
        pose: Mapping with xyz position and xyzw quaternion components.

    Returns:
        Position array of shape ``(N, 3)`` and quaternion array of shape ``(N, 4)``.
    """
    pos = np.column_stack((pose["x"], pose["y"], pose["z"]))
    quat_xyzw = np.column_stack((pose["qx"], pose["qy"], pose["qz"], pose["qw"]))
    return pos, quat_xyzw


def arrays_to_pose(pos: np.ndarray, quat_xyzw: np.ndarray) -> dict[str, np.ndarray]:
    """Convert position and quaternion arrays to a pose mapping.

    Args:
        pos: Position array of shape ``(N, 3)``.
        quat_xyzw: Quaternion array of shape ``(N, 4)`` in xyzw order.

    Returns:
        Mapping from pose component name to array.
    """
    return {
        "x": pos[:, 0],
        "y": pos[:, 1],
        "z": pos[:, 2],
        "qx": quat_xyzw[:, 0],
        "qy": quat_xyzw[:, 1],
        "qz": quat_xyzw[:, 2],
        "qw": quat_xyzw[:, 3],
    }


def slice_pose(pose: dict[str, np.ndarray], stop: int) -> dict[str, np.ndarray]:
    """Slice all pose components up to a stop index.

    Args:
        pose: Mapping from pose component name to array.
        stop: Exclusive stop index.

    Returns:
        Sliced pose mapping.
    """
    return {name: values[:stop] for name, values in pose.items()}


def pose_positions(pose: dict[str, np.ndarray]) -> np.ndarray:
    """Extract xyz positions from a pose mapping.

    Args:
        pose: Mapping containing ``x``, ``y``, and ``z`` arrays.

    Returns:
        Position array of shape ``(N, 3)``.
    """
    return np.column_stack((pose["x"], pose["y"], pose["z"]))


def resample_pose_for_plot(pose: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Resample a pose sequence to equal time steps for component plotting.

    Args:
        pose: Mapping with xyz position and xyzw quaternion components.

    Returns:
        Resampled pose mapping using the same component order as the input.
    """
    pos, quat_xyzw = poses_to_arrays(pose)
    quat_wxyz = np.column_stack((quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]))

    pos_resampled, quat_wxyz_resampled = resample_trajectory_6d_equal_dt(
        pos,
        quat_wxyz,
        sample_hz=SAMPLE_HZ,
        speed=DEFAULT_SPEED * PLOT_RESAMPLE_SPEED_SCALE,
    )
    quat_xyzw_resampled = np.column_stack(
        (
            quat_wxyz_resampled[:, 1],
            quat_wxyz_resampled[:, 2],
            quat_wxyz_resampled[:, 3],
            quat_wxyz_resampled[:, 0],
        )
    )
    return arrays_to_pose(pos_resampled, make_quaternions_continuous(quat_xyzw_resampled))


def companion_3d_output_path(output: Path | None) -> Path | None:
    """Build the companion 3D plot output path.

    Args:
        output: Main component plot output path, or ``None``.

    Returns:
        Companion output path with ``_3d`` appended to the stem, or ``None``.
    """
    if output is None:
        return None
    suffix = output.suffix or ".png"
    return output.with_name(f"{output.stem}_3d{suffix}")


def set_equal_3d_axes(ax, points: np.ndarray, padding: float = 0.08) -> None:
    """Set equal 3D axis ranges around the provided points.

    Args:
        ax: Matplotlib 3D axis to update.
        points: Array of shape ``(N, 3)`` used to determine axis bounds.
        padding: Fractional padding added around the bounding radius.
    """
    if len(points) == 0:
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius * (1.0 + padding), 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to a rotation matrix.

    Args:
        q_xyzw: Quaternion in xyzw order.

    Returns:
        Rotation matrix of shape ``(3, 3)``.
    """
    x, y, z, w = q_xyzw
    n = np.linalg.norm(q_xyzw)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = q_xyzw / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a normalized xyzw quaternion.

    Args:
        R: Rotation matrix of shape ``(3, 3)``.

    Returns:
        Quaternion in xyzw order.
    """
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.asarray([x, y, z, w], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def make_quaternions_continuous(quat_xyzw: np.ndarray) -> np.ndarray:
    """Flip quaternion signs to keep adjacent samples continuous.

    Args:
        quat_xyzw: Quaternion array of shape ``(N, 4)`` in xyzw order.

    Returns:
        Copy of the quaternion array with sign flips applied when needed.
    """
    out = np.asarray(quat_xyzw, dtype=np.float64).copy()
    for i in range(1, len(out)):
        if np.dot(out[i - 1], out[i]) < 0.0:
            out[i] *= -1.0
    return out


def transform_pose_to_ref(
    pose: dict[str, np.ndarray],
    R_ref_to_world: np.ndarray,
    scale: float,
    translation: np.ndarray,
) -> dict[str, np.ndarray]:
    """Transform a world-frame pose sequence into the reference frame.

    Args:
        pose: Pose mapping in the world frame.
        R_ref_to_world: Rotation matrix from reference frame to world frame.
        scale: Similarity-transform scale.
        translation: Similarity-transform translation vector.

    Returns:
        Pose mapping expressed in the reference frame.
    """
    pos_world, quat_world_xyzw = poses_to_arrays(pose)

    pos_ref = ((pos_world - translation) / scale) @ R_ref_to_world
    quat_ref_xyzw = np.empty_like(quat_world_xyzw)
    R_world_to_ref = R_ref_to_world.T

    for i, q_xyzw in enumerate(quat_world_xyzw):
        R_world_body = quat_xyzw_to_rotmat(q_xyzw)
        R_ref_body = R_world_to_ref @ R_world_body
        quat_ref_xyzw[i] = rotmat_to_quat_xyzw(R_ref_body)

    return arrays_to_pose(pos_ref, make_quaternions_continuous(quat_ref_xyzw))


def latest_file(directory: Path, prefixes: tuple[str, ...], suffix: str) -> Path:
    """Find the newest file matching any prefix and suffix.

    Args:
        directory: Directory to search.
        prefixes: Accepted filename prefixes.
        suffix: Required filename suffix.

    Returns:
        Most recently modified matching file.

    Raises:
        FileNotFoundError: If no matching file exists.
    """
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix == suffix and any(path.name.startswith(prefix) for prefix in prefixes)
    ]
    if not candidates:
        raise FileNotFoundError(f"No {suffix} file with prefixes {prefixes} in {directory}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def infer_ref_root(sample_dir: Path) -> Path | None:
    """Infer the reference root from a prediction sample path.

    Args:
        sample_dir: Prediction sample directory.

    Returns:
        Inferred reference root, or ``None`` when the path does not contain
        a ``preds`` segment.
    """
    parts = sample_dir.parts
    if "preds" not in parts:
        return None

    idx = parts.index("preds")
    return Path(*parts[:idx]) / "refs" / "processed"


def resolve_reference_file(sample_dir: Path, transform_data: dict, ref_root: Path | None) -> Path:
    """Resolve the matched reference CSV for a sample directory.

    Args:
        sample_dir: Directory containing prompt, prediction, and transform files.
        transform_data: Loaded transform metadata.
        ref_root: Optional root containing processed reference CSVs.

    Returns:
        Path to the matched reference CSV.

    Raises:
        FileNotFoundError: If no reference CSV can be found or inferred.
    """
    candidates = [
        path
        for path in sample_dir.iterdir()
        if path.is_file()
        and path.suffix == ".csv"
        and any(path.name.startswith(prefix) for prefix in REF_PREFIXES)
    ]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    skill_name = transform_data.get("skill_name")
    if not skill_name:
        raise FileNotFoundError(
            f"No matched reference CSV in {sample_dir}, and transform JSON has no skill_name"
        )

    base_ref_root = ref_root if ref_root is not None else infer_ref_root(sample_dir)
    if base_ref_root is None:
        raise FileNotFoundError(
            f"No matched reference CSV in {sample_dir}; pass --ref-root to resolve skill {skill_name}"
        )

    skill_group = skill_name.rsplit("_", 1)[0]
    ref_path = base_ref_root / skill_group / f"{skill_name}.csv"
    if ref_path.exists():
        return ref_path

    flat_ref_path = base_ref_root / f"{skill_name}.csv"
    if flat_ref_path.exists():
        return flat_ref_path

    raise FileNotFoundError(
        f"Reference file not found for skill {skill_name}: {ref_path} or {flat_ref_path}"
    )
    return ref_path


def find_sample_dirs(root: Path) -> list[Path]:
    """Find complete sample directories under a root.

    Args:
        root: Directory to search.

    Returns:
        Sorted list of directories containing prompt CSV, prediction CSV, and
        similarity-transform JSON files.

    Raises:
        ValueError: If ``root`` is a file.
    """
    if root.is_file():
        raise ValueError(f"Expected a directory, got file: {root}")

    sample_dirs = []
    for directory in [root, *root.rglob("*")]:
        if not directory.is_dir():
            continue
        names = [path.name for path in directory.iterdir() if path.is_file()]
        has_prompt = any(name.startswith(PROMPT_PREFIX) and name.endswith(".csv") for name in names)
        has_pred = any(name.startswith(PRED_PREFIX) and name.endswith(".csv") for name in names)
        has_transform = any(name.startswith(TRANSFORM_PREFIX) and name.endswith(".json") for name in names)
        if has_prompt and has_pred and has_transform:
            sample_dirs.append(directory)
    return sorted(sample_dirs)


def resolve_sample_dirs(root: Path, sample: str | None, plot_all: bool) -> list[Path]:
    """Resolve which sample directories should be plotted.

    Args:
        root: Sample directory or prediction root directory.
        sample: Optional sample path relative to ``root``.
        plot_all: Whether to plot every complete sample under ``root``.

    Returns:
        List of sample directories to plot.

    Raises:
        FileNotFoundError: If requested sample directories cannot be found.
        ValueError: If multiple samples exist and no selection mode is provided.
    """
    if sample:
        directory = root / sample
        if not directory.exists():
            raise FileNotFoundError(f"Sample directory does not exist: {directory}")
        return [directory]

    sample_dirs = find_sample_dirs(root)
    if not sample_dirs:
        raise FileNotFoundError(f"No complete sample directories found under {root}")
    if plot_all or sample_dirs == [root]:
        return sample_dirs

    examples = "\n".join(f"  {path.relative_to(root)}" for path in sample_dirs[:12])
    more = "" if len(sample_dirs) <= 12 else f"\n  ... {len(sample_dirs) - 12} more"
    raise ValueError(
        f"{root} contains {len(sample_dirs)} sample directories. "
        f"Pass --sample, point DATA_DIR at one sample directory, or use --all.\n{examples}{more}"
    )


def plot_sample(sample_dir: Path, output: Path | None, show: bool, ref_root: Path | None) -> None:
    """Plot aligned component and 3D trajectory figures for one sample.

    Args:
        sample_dir: Directory containing one complete prediction sample.
        output: Optional output path for the component figure.
        show: Whether to display figures interactively.
        ref_root: Optional root containing processed reference CSVs.
    """
    prompt_path = latest_file(sample_dir, (PROMPT_PREFIX,), ".csv")
    pred_path = latest_file(sample_dir, (PRED_PREFIX,), ".csv")
    transform_path = latest_file(sample_dir, (TRANSFORM_PREFIX,), ".json")

    R, scale, translation, transform_data = load_similarity_transform(transform_path)
    ref_path = resolve_reference_file(sample_dir, transform_data, ref_root)

    ref = load_pose_csv(ref_path)
    ref_quat = np.column_stack((ref["qx"], ref["qy"], ref["qz"], ref["qw"]))
    ref.update(arrays_to_pose(np.column_stack((ref["x"], ref["y"], ref["z"])), make_quaternions_continuous(ref_quat)))

    prompt_ref = transform_pose_to_ref(load_pose_csv(prompt_path), R, scale, translation)
    pred_ref = transform_pose_to_ref(load_pose_csv(pred_path), R, scale, translation)
    pred_start_idx = len(prompt_ref["x"])
    pred_stop = max(0, len(ref["x"]) + PRED_EXTRA_POINTS_AFTER_REF - pred_start_idx)
    pred_ref = slice_pose(pred_ref, pred_stop)

    skill_label = transform_data.get("skill_name", ref_path.stem)

    fig, axes = plt.subplots(len(COMPONENTS), 1, figsize=(13, 14), sharex=False)
    fig.suptitle(f"{sample_dir.name}: {skill_label} reference frame components", fontsize=15)

    series = (
        ("ref", ref, "tab:blue"),
        ("prompt aligned", prompt_ref, "tab:orange"),
        ("prediction aligned", pred_ref, "tab:green"),
    )
    series_2d = tuple((label, resample_pose_for_plot(pose), color) for label, pose, color in series)
    pred_start_idx_2d = len(series_2d[1][1]["x"])

    for ax, component in zip(axes, COMPONENTS):
        for label, pose, color in series_2d:
            start_idx = pred_start_idx_2d if label == "prediction aligned" else 0
            x_idx = start_idx + np.arange(len(pose[component]))
            ax.plot(x_idx, pose[component], label=label, color=color, linewidth=1.4)
        ax.set_ylabel(component)
        ax.grid(True, alpha=0.3)
        if component == COMPONENTS[0]:
            ax.legend(loc="best")

    axes[-1].set_xlabel("point index")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"saved {output}")

    fig_3d = plt.figure(figsize=(8, 7))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.set_title(f"{sample_dir.name}: {skill_label} aligned positions")

    for label, pose, color in series:
        pos = pose_positions(pose)
        if len(pos) == 0:
            continue
        linewidth = 3.0 if label == "prediction aligned" else 1.8
        ax_3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=label, color=color, linewidth=linewidth)

    all_points = np.vstack([pose_positions(pose) for _, pose, _ in series if len(pose["x"]) > 0])
    set_equal_3d_axes(ax_3d, all_points)
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.legend(loc="best")
    ax_3d.grid(True, alpha=0.3)
    fig_3d.tight_layout()

    output_3d = companion_3d_output_path(output)
    if output_3d is not None:
        output_3d.parent.mkdir(parents=True, exist_ok=True)
        fig_3d.savefig(output_3d, dpi=200)
        print(f"saved {output_3d}")

    if show:
        plt.show()
    plt.close(fig)
    plt.close(fig_3d)


def main() -> None:
    """Parse command-line arguments and plot selected samples."""
    parser = argparse.ArgumentParser(
        description=(
            "Transform prompt/prediction trajectories into the matched reference frame "
            "and plot x,y,z,qx,qy,qz,qw against point index."
        )
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data/06-02/preds",
        type=Path,
        help="A sample directory, or the preds root when used with --sample/--all.",
    )
    parser.add_argument("--sample", help="Sample path relative to data_dir, for example line2/line2_1.")
    parser.add_argument("--all", action="store_true", help="Plot every complete sample directory under data_dir.")
    parser.add_argument(
        "--ref-root",
        type=Path,
        help="Reference root containing processed skill CSVs, for example data/06-02/refs/processed.",
    )
    parser.add_argument("--output", type=Path, help="Output PNG path. For --all, this is treated as an output directory.")
    parser.add_argument("--show", action="store_true", help="Show the figure window after saving/creating it.")
    args = parser.parse_args()

    sample_dirs = resolve_sample_dirs(args.data_dir, args.sample, args.all)
    for sample_dir in sample_dirs:
        if args.output is None:
            output = None if args.show else sample_dir / "aligned_components.png"
        elif len(sample_dirs) == 1:
            output = args.output
        else:
            output = args.output / f"{sample_dir.parent.name}_{sample_dir.name}_aligned_components.png"

        plot_sample(sample_dir, output, show=args.show, ref_root=args.ref_root)


if __name__ == "__main__":
    main()
