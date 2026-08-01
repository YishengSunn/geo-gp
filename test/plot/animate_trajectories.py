"""Animate prompt, leader, prediction, and fused trajectories by timestamp."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from scipy.spatial.transform import Rotation, Slerp


DEFAULT_INPUT_DIR = Path("data/06-02/preds/arc/arc_1")
DEFAULT_OUTPUT = Path("data/06-02/preds/arc/arc_1/arc_1_trajectory_animation.html")


@dataclass(frozen=True)
class TrajectorySpec:
    """Configuration for locating and rendering one trajectory.

    Attributes:
        name: Human-readable trajectory name.
        pattern: Glob pattern used to find the trajectory CSV.
        color: Matplotlib color used for the trajectory.
        linewidth: Line width used for the trajectory.
        marker: Marker used for the current trajectory position.
    """

    name: str
    pattern: str
    color: str
    linewidth: float
    marker: str


SPECS = (
    TrajectorySpec("prompt", "prompt*.csv", "#3b82f6", 1.8, "o"),
    TrajectorySpec("leader", "leader*.csv", "#f97316", 2.0, "^"),
    TrajectorySpec("prediction", "prediction*.csv", "#10b981", 1.8, "s"),
    TrajectorySpec("fused", "fused*.csv", "#111827", 2.5, "D"),
)


@dataclass
class Trajectory:
    """Loaded trajectory data and interpolation helpers.

    Attributes:
        spec: Rendering and file-matching configuration.
        path: CSV path used to load the trajectory.
        time: Timestamp array.
        xyz: Position array of shape ``(N, 3)``.
        quat: Quaternion array of shape ``(N, 4)`` in xyzw order.
        alpha: Optional alpha values loaded from the CSV.
        slerp: Optional spherical interpolation object for orientations.
    """

    spec: TrajectorySpec
    path: Path
    time: np.ndarray
    xyz: np.ndarray
    quat: np.ndarray
    alpha: np.ndarray | None
    slerp: Slerp | None

    @property
    def start(self) -> float:
        """Return the first trajectory timestamp.

        Returns:
            First timestamp as a float.
        """
        return float(self.time[0])

    @property
    def end(self) -> float:
        """Return the last trajectory timestamp.

        Returns:
            Last timestamp as a float.
        """
        return float(self.time[-1])

    def active(self, t: float) -> bool:
        """Check whether a timestamp lies inside the trajectory time span.

        Args:
            t: Timestamp to check.

        Returns:
            ``True`` if ``t`` is between the first and last timestamps.
        """
        return self.start <= t <= self.end

    def position_at(self, t: float) -> np.ndarray | None:
        """Interpolate the trajectory position at a timestamp.

        Args:
            t: Timestamp at which to evaluate the trajectory.

        Returns:
            Interpolated xyz position, or ``None`` if ``t`` is outside the
            trajectory time span.
        """
        if not self.active(t):
            return None
        return np.array([np.interp(t, self.time, self.xyz[:, axis]) for axis in range(3)])

    def rotation_at(self, t: float) -> Rotation | None:
        """Interpolate the trajectory orientation at a timestamp.

        Args:
            t: Timestamp at which to evaluate the orientation.

        Returns:
            Interpolated rotation, or ``None`` if ``t`` is outside the
            trajectory time span or orientation interpolation is unavailable.
        """
        if not self.active(t) or self.slerp is None:
            return None
        return self.slerp([t])[0]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create a timestamp-aligned animation for arc trajectory CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing trajectory CSV files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output animation path (.html, .mp4, or .gif). Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Animation frame rate for saved output.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Approximate animation duration in seconds.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Figure DPI used for embedded HTML frames.",
    )
    parser.add_argument(
        "--trail",
        type=float,
        default=0.0,
        help="Trail length in seconds. Use 0 to show each full trajectory up to current time.",
    )
    parser.add_argument(
        "--no-orientation",
        action="store_true",
        help="Hide quaternion orientation triads at current trajectory positions.",
    )
    parser.add_argument(
        "--view",
        nargs=2,
        type=float,
        default=(24.0, -56.0),
        metavar=("ELEV", "AZIM"),
        help="3D view elevation and azimuth in degrees.",
    )
    return parser.parse_args()


def first_match(input_dir: Path, pattern: str) -> Path:
    """Return the latest lexicographic file matching a pattern.

    Args:
        input_dir: Directory to search.
        pattern: Glob pattern to match.

    Returns:
        Last sorted matching path.

    Raises:
        FileNotFoundError: If no matching file exists.
    """
    matches = sorted(input_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {input_dir}")
    return matches[-1]


def load_trajectory(input_dir: Path, spec: TrajectorySpec) -> Trajectory:
    """Load a trajectory CSV and build interpolation helpers.

    Args:
        input_dir: Directory containing trajectory CSV files.
        spec: Trajectory specification defining the filename pattern.

    Returns:
        Loaded trajectory object.

    Raises:
        FileNotFoundError: If no file matches ``spec.pattern``.
        ValueError: If the CSV is missing required columns.
    """
    path = first_match(input_dir, spec.pattern)
    df = pd.read_csv(path)
    required = ("time", "x", "y", "z", "qx", "qy", "qz", "qw")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    time = df["time"].to_numpy(dtype=float)
    xyz = df[["x", "y", "z"]].to_numpy(dtype=float)
    quat = df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float)
    alpha = df["alpha"].to_numpy(dtype=float) if "alpha" in df.columns else None

    slerp = None
    if len(time) >= 2:
        slerp = Slerp(time, Rotation.from_quat(quat))

    return Trajectory(spec, path, time, xyz, quat, alpha, slerp)


def make_frame_times(trajectories: list[Trajectory], fps: int, duration: float) -> np.ndarray:
    """Create animation frame timestamps.

    Args:
        trajectories: Loaded trajectories to animate.
        fps: Target frames per second.
        duration: Approximate animation duration in seconds.

    Returns:
        Array of elapsed animation times.
    """
    prompt = next(traj for traj in trajectories if traj.spec.name == "prompt")
    prompt_span = prompt.end - prompt.start
    other_spans = [traj.end - traj.start for traj in trajectories if traj.spec.name != "prompt"]
    total_span = prompt_span + max(other_spans, default=0.0)
    frame_count = max(2, int(round(fps * duration)))
    return np.linspace(0.0, total_span, frame_count)


def local_time_for_animation(
    traj: Trajectory,
    elapsed_time: float,
    prompt_span: float,
) -> float | None:
    """Map elapsed animation time to a trajectory-local timestamp.

    The prompt is shown first. Other trajectories start after the prompt phase
    completes.

    Args:
        traj: Trajectory being evaluated.
        elapsed_time: Elapsed animation time.
        prompt_span: Duration of the prompt trajectory.

    Returns:
        Local timestamp for ``traj``, or ``None`` if the trajectory should not
        be visible yet.
    """
    if traj.spec.name == "prompt":
        return min(traj.start + elapsed_time, traj.end)

    if elapsed_time < prompt_span:
        return None

    return min(traj.start + elapsed_time - prompt_span, traj.end)


def set_axes_equal(ax: plt.Axes, trajectories: list[Trajectory]) -> None:
    """Set equal 3D axis limits around all trajectory points.

    Args:
        ax: Matplotlib 3D axis to update.
        trajectories: Trajectories whose points define the axis bounds.
    """
    points = np.vstack([traj.xyz for traj in trajectories])
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0)
    radius = max(radius, 0.01)

    labels = ("x [m]", "y [m]", "z [m]")
    setters = (ax.set_xlim, ax.set_ylim, ax.set_zlim)
    for axis, setter in enumerate(setters):
        setter(center[axis] - radius, center[axis] + radius)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])


def draw_orientation(
    ax: plt.Axes,
    origin: np.ndarray,
    rotation: Rotation,
    length: float,
    artists: list,
) -> None:
    """Draw a local orientation triad at a point.

    Args:
        ax: Matplotlib 3D axis to draw on.
        origin: Triad origin in xyz coordinates.
        rotation: Rotation defining the triad orientation.
        length: Axis length for the triad arrows.
        artists: Mutable list that receives created Matplotlib artists.
    """
    basis = rotation.apply(np.eye(3)) * length
    colors = ("#ef4444", "#22c55e", "#3b82f6")
    for axis in range(3):
        artists.append(
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                basis[axis, 0],
                basis[axis, 1],
                basis[axis, 2],
                color=colors[axis],
                linewidth=1.2,
                arrow_length_ratio=0.25,
            )
        )


def build_animation(
    trajectories: list[Trajectory],
    frame_times: np.ndarray,
    fps: int,
    trail: float,
    show_orientation: bool,
    view: tuple[float, float],
    dpi: int,
) -> animation.FuncAnimation:
    """Build the timestamp-aligned trajectory animation.

    Args:
        trajectories: Loaded trajectories to animate.
        frame_times: Elapsed animation times for all frames.
        fps: Target frames per second.
        trail: Trail length in seconds, or ``0`` to show the full history.
        show_orientation: Whether to draw current orientation triads.
        view: 3D view elevation and azimuth in degrees.
        dpi: Figure DPI.

    Returns:
        Matplotlib animation object.
    """
    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=view[0], azim=view[1])
    set_axes_equal(ax, trajectories)
    ax.grid(True, alpha=0.28)
    ax.set_title("Timestamp-aligned trajectory animation")

    prompt_traj = next(traj for traj in trajectories if traj.spec.name == "prompt")
    leader_traj = next(traj for traj in trajectories if traj.spec.name == "leader")

    full_lines = {}
    trail_lines = {}
    current_points = {}
    for traj in trajectories:
        reference_xyz = traj.xyz
        if traj.spec.name == "leader":
            reference_xyz = np.vstack([prompt_traj.xyz[-1], traj.xyz])

        (full_line,) = ax.plot(
            reference_xyz[:, 0],
            reference_xyz[:, 1],
            reference_xyz[:, 2],
            color=traj.spec.color,
            linewidth=traj.spec.linewidth,
            alpha=0.18,
            label=f"{traj.spec.name} ({traj.path.name})",
        )
        (trail_line,) = ax.plot(
            [],
            [],
            [],
            color=traj.spec.color,
            linewidth=traj.spec.linewidth,
            alpha=0.95,
        )
        (point,) = ax.plot(
            [],
            [],
            [],
            marker=traj.spec.marker,
            color=traj.spec.color,
            markersize=7,
            linestyle="None",
        )
        full_lines[traj.spec.name] = full_line
        trail_lines[traj.spec.name] = trail_line
        current_points[traj.spec.name] = point


    time_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes)
    ax.legend(loc="upper right", fontsize=8)
    orientation_artists: list = []

    all_points = np.vstack([traj.xyz for traj in trajectories])
    orientation_length = float((all_points.max(axis=0) - all_points.min(axis=0)).max() * 0.07)
    orientation_length = max(orientation_length, 0.01)
    prompt = next(traj for traj in trajectories if traj.spec.name == "prompt")
    prompt_span = prompt.end - prompt.start

    def update(frame_idx: int):
        """Update artists for one animation frame.

        Args:
            frame_idx: Index into ``frame_times``.

        Returns:
            Artists updated for the current frame.
        """
        for artist in orientation_artists:
            artist.remove()
        orientation_artists.clear()

        elapsed_time = float(frame_times[frame_idx])
        phase = "prompt" if elapsed_time <= prompt_span else "execution / prediction / fusion"
        for traj in trajectories:
            local_time = local_time_for_animation(traj, elapsed_time, prompt_span)
            full_lines[traj.spec.name].set_visible(local_time is not None)

            trail_line = trail_lines[traj.spec.name]
            point = current_points[traj.spec.name]
            if local_time is None:
                trail_line.set_data([], [])
                trail_line.set_3d_properties([])
                point.set_data([], [])
                point.set_3d_properties([])
                continue

            if trail > 0:
                mask = (traj.time >= max(traj.start, local_time - trail)) & (traj.time <= local_time)
            else:
                mask = traj.time <= local_time

            trail_xyz = traj.xyz[mask]
            if traj.spec.name == "leader" and len(trail_xyz) > 0:
                trail_xyz = np.vstack([prompt_traj.xyz[-1], trail_xyz])

            trail_line.set_data(trail_xyz[:, 0], trail_xyz[:, 1])
            trail_line.set_3d_properties(trail_xyz[:, 2])

            position = traj.position_at(local_time)
            if position is None:
                point.set_data([], [])
                point.set_3d_properties([])
                continue

            point.set_data([position[0]], [position[1]])
            point.set_3d_properties([position[2]])

            rotation = traj.rotation_at(local_time)
            if show_orientation and rotation is not None:
                draw_orientation(ax, position, rotation, orientation_length, orientation_artists)

        time_text.set_text(f"animation = {elapsed_time:.3f} s\nphase = {phase}")
        return (
            list(full_lines.values())
            + list(trail_lines.values())
            + list(current_points.values())
            + [time_text]
            + orientation_artists
        )

    interval_ms = 1000.0 / fps
    return animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_times),
        interval=interval_ms,
        blit=False,
    )



def clean_jshtml_controls(html: str) -> str:
    """Remove selected controls from Matplotlib JSHTML output.

    Args:
        html: Raw JSHTML animation string.

    Returns:
        HTML string with speed and reflect controls removed.
    """
    html = re.sub(
        r'\s*<button title="Decrease speed".*?</button>',
        "",
        html,
        flags=re.DOTALL,
    )
    html = re.sub(
        r'\s*<button title="Increase speed".*?</button>',
        "",
        html,
        flags=re.DOTALL,
    )
    html = re.sub(
        r'\s*<input(?=[^>]*type="radio")(?=[^>]*value="reflect")[^>]*>'
        r'\s*<label[^>]*>Reflect</label>',
        "",
        html,
        flags=re.DOTALL,
    )
    return html


def save_animation(anim: animation.FuncAnimation, output: Path, fps: int) -> None:
    """Save an animation to HTML, GIF, or MP4.

    Args:
        anim: Animation object to save.
        output: Output file path.
        fps: Frames per second for encoded output.

    Raises:
        ValueError: If the output suffix is unsupported.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".html":
        plt.rcParams["animation.embed_limit"] = 200.0
        html = clean_jshtml_controls(anim.to_jshtml(fps=fps, default_mode="loop"))
        output.write_text(html, encoding="utf-8")
    elif suffix == ".gif":
        anim.save(output, writer="pillow", fps=fps)
    elif suffix == ".mp4":
        anim.save(output, writer="ffmpeg", fps=fps)
    else:
        raise ValueError("Output suffix must be .html, .gif, or .mp4")


def main() -> None:
    """Run the trajectory animation CLI."""
    args = parse_args()
    trajectories = [load_trajectory(args.input_dir, spec) for spec in SPECS]
    frame_times = make_frame_times(trajectories, args.fps, args.duration)
    anim = build_animation(
        trajectories=trajectories,
        frame_times=frame_times,
        fps=args.fps,
        trail=args.trail,
        show_orientation=not args.no_orientation,
        view=tuple(args.view),
        dpi=args.dpi,
    )
    save_animation(anim, args.output, args.fps)
    print(f"Saved animation to {args.output}")


if __name__ == "__main__":
    main()
