import numpy as np
import matplotlib.pyplot as plt

from geometry.angles import angles_relative_to_start_tangent


def plot_angle_changes(ref_pts, probe_pts, k_hist=10, min_r=1e-3):
    """
    Plot relative angle changes of reference and probe trajectories.

    Args:
        ref_pts: list or numpy array of shape (N, 2)
        probe_pts: list or numpy array of shape (M, 2)
        k_hist: int, number of points to estimate start tangent
        min_r: float, minimum radius to consider valid angle

    Returns:
        ref_angles: numpy array of shape (N,), relative angles of reference trajectory
        probe_angles: numpy array of shape (M,), relative angles of probe trajectory
    """
    # --- Calculate angle sequences ---
    ref_angles, _ = angles_relative_to_start_tangent(ref_pts, k_hist, min_r)
    probe_angles, _ = angles_relative_to_start_tangent(probe_pts, k_hist, min_r)

    # --- Plotting ---
    plt.figure(figsize=(10,4))
    plt.plot(ref_angles, label="Reference traj (relative angle)", color='red')
    plt.plot(probe_angles, label="Probe traj (relative angle)", color='blue')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel("Point Index")
    plt.ylabel("Relative Angle (Rad)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Relative Angle Changes of Reference vs Probe")
    plt.show()

    return ref_angles, probe_angles

def plot_series_and_mse(
    sampled,
    probe_pts,
    pred_scaled,
    baseline_preds,
    dtheta,
    scale,
    sample_hz,
):
    """
    Plot the series of reference trajectory (transformed to probe frame), probe trajectory,
    predicted trajectory (probe-based), baseline trajectory, and their point-wise MSE over time.

    Three plots (all aligned in the probe coordinate system) + export CSV:
    1) X: ref(→probe), probe, pred(probe), baseline
    2) Y: ref(→probe), probe, pred(probe), baseline
    3) Point-wise MSE (starting from idx_start=len(probe)):
    - MSE(pred(probe) vs ref_in_probe)
    - MSE(baseline vs ref_in_probe)
    Also write all the above data used for plotting into a CSV, with columns aligned based on the global index.

    Args:
        sampled: torch tensor of shape (Nr, 2), sampled reference trajectory
        probe_pts: list or numpy array of shape (Np, 2), probe trajectory points
        pred_scaled: numpy array of shape (Hp, 2), predicted trajectory based on probe
        baseline_preds: numpy array of shape (Hb, 2), baseline predicted trajectory
        dtheta: float, rotation angle from reference to probe frame (radians)
        scale: float, scaling factor from reference to probe frame
        sample_hz: float, sampling frequency (not used in current plotting)
    """
    print("Generating series and MSE plots...")

    # --- Basic checks ---
    if sampled is None or len(sampled) == 0:
        print("No reference trajectory sampled (sampled)!")
        print()
        return
    
    if probe_pts is None or len(probe_pts) < 1:
        print("No probe trajectory!")
        print()
        return
    
    # --- Data extraction ---
    ref_np = sampled.detach().cpu().numpy().astype(np.float64)  # Shape: (Nr, 2)
    probe_np = np.asarray(probe_pts, dtype=np.float64)          # Shape: (Np, 2)
    pred_probe_world = (
        np.asarray(pred_scaled, dtype=np.float64)
        if pred_scaled is not None and len(pred_scaled) > 0
        else np.zeros((0, 2), dtype=np.float64)
    )
    baseline_world = (
        np.asarray(baseline_preds, dtype=np.float64)
        if baseline_preds is not None and len(baseline_preds) > 0
        else np.zeros((0, 2), dtype=np.float64)
    )

    # --- Manual parameters (forward transform from ref to probe) ---
    dtheta = float(dtheta)
    scale  = float(scale)
    c, s = np.cos(dtheta), np.sin(dtheta)
    R_fwd = np.array([[c, -s], [s, c]], dtype=np.float64)

    ref_origin = ref_np[0]
    probe_origin = probe_np[0]

    # Ref → Probe：preds_new = scale * (R @ (ref - ref0)) + probe0
    ref_center = ref_np - ref_origin
    ref_rot = (R_fwd @ ref_center.T).T
    ref_in_probe = scale * ref_rot + probe_origin   # Shape: (Nr, 2)

    # --- Index alignment: predictions start from the probe end index ---
    idx_start = len(probe_np)

    # --- Time axes ---
    t_probe      = np.arange(len(probe_np))
    t_ref_probe  = np.arange(len(ref_in_probe))
    t_pred_probe = np.arange(idx_start, idx_start + len(pred_probe_world))
    t_base_probe = np.arange(idx_start, idx_start + len(baseline_world))

    # --- Point-wise MSE (aligned with ref_in_probe, starting from idx_start) ---
    def aligned_mse(pred_xy, ref_xy, start_idx):
        """
        Calculate point-wise MSE between pred_xy and ref_xy[start_idx:].

        Args:
            pred_xy: (L, 2) np.ndarray of predicted points
            ref_xy: (N, 2) np.ndarray of reference points
            start_idx: int, starting index in ref_xy for alignment

        Returns:
            mse: (M,) np.ndarray of point-wise MSE, where M = min(len(pred_xy), len(ref_xy) - start_idx)
        """
        if len(pred_xy) == 0:
            return np.zeros(0, dtype=np.float64)
        
        max_len = max(0, len(ref_xy) - start_idx)
        L = int(min(len(pred_xy), max_len))
        if L <= 0:
            return np.zeros(0, dtype=np.float64)
        
        diff = pred_xy[:L] - ref_xy[start_idx:start_idx+L]

        return np.mean(diff**2, axis=1)  # (L,)

    mse_probe = aligned_mse(pred_probe_world, ref_in_probe, idx_start)
    mse_base  = aligned_mse(baseline_world,   ref_in_probe, idx_start)

    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

    # 1) X
    ax = axes[0]
    ax.plot(t_ref_probe, ref_in_probe[:, 0], label="ref→probe_x", linewidth=2)
    ax.plot(t_probe, probe_np[:, 0], label="probe_x", linestyle="--")
    if len(pred_probe_world) > 0:
        ax.plot(t_pred_probe, pred_probe_world[:, 0], label="pred(probe)_x", linestyle="-.")
    if len(baseline_world) > 0:
        ax.plot(t_base_probe, baseline_world[:, 0], label="baseline_x", linestyle=":")
    ax.set_title("X components in PROBE frame (pred starts at probe end index)")
    ax.set_xlabel("Index"); ax.set_ylabel("x")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")

    # 2) Y
    ax = axes[1]
    ax.plot(t_ref_probe, ref_in_probe[:, 1], label="ref→probe_y", linewidth=2)
    ax.plot(t_probe, probe_np[:, 1], label="probe_y", linestyle="--")
    if len(pred_probe_world) > 0:
        ax.plot(t_pred_probe, pred_probe_world[:, 1], label="pred(probe)_y", linestyle="-.")
    if len(baseline_world) > 0:
        ax.plot(t_base_probe, baseline_world[:, 1], label="baseline_y", linestyle=":")
    ax.set_title("Y components in PROBE frame (pred starts at probe end index)")
    ax.set_xlabel("Index"); ax.set_ylabel("y")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")

    # 3) MSE
    ax = axes[2]
    if len(mse_probe) > 0:
        ax.plot(np.arange(idx_start, idx_start + len(mse_probe)), mse_probe,
                label="MSE(pred(probe) vs ref→probe)")
    if len(mse_base) > 0:
        ax.plot(np.arange(idx_start, idx_start + len(mse_base)), mse_base,
                label="MSE(baseline vs ref→probe)")
    ax.set_title("Per-step MSE vs ref→probe (aligned at probe end index)")
    ax.set_xlabel("Index"); ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")

    plt.show()

    # ---------------- CSV export (aligned by global index) ----------------
    import csv
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"series_mse_{ts}.csv"

    # Construct global index range
    max_index = max(
        len(ref_in_probe) - 1,
        len(probe_np) - 1,
        (idx_start + len(pred_probe_world) - 1) if len(pred_probe_world) > 0 else -1,
        (idx_start + len(baseline_world) - 1) if len(baseline_world) > 0 else -1
    )
    max_index = int(max(0, max_index))

    # Precompute MSE mapping to global index
    mse_probe_map = {}
    for i in range(len(mse_probe)):
        mse_probe_map[idx_start + i] = mse_probe[i]
    mse_base_map = {}
    for i in range(len(mse_base)):
        mse_base_map[idx_start + i] = mse_base[i]

    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        # Header
        w.writerow([
            "index",
            "ref_to_probe_x", "ref_to_probe_y",
            "probe_x", "probe_y",
            "pred_probe_x", "pred_probe_y",
            "baseline_x", "baseline_y",
            "mse_pred_vs_ref_to_probe", "mse_baseline_vs_ref_to_probe"
        ])

        for idx in range(max_index + 1):
            row = [idx]
            # Ref → Probe
            if idx < len(ref_in_probe):
                row += [ref_in_probe[idx, 0], ref_in_probe[idx, 1]]
            else:
                row += ["", ""]
            # Probe
            if idx < len(probe_np):
                row += [probe_np[idx, 0], probe_np[idx, 1]]
            else:
                row += ["", ""]
            # Pred(probe)
            p_i = idx - idx_start
            if 0 <= p_i < len(pred_probe_world):
                row += [pred_probe_world[p_i, 0], pred_probe_world[p_i, 1]]
            else:
                row += ["", ""]
            # Baseline
            b_i = idx - idx_start
            if 0 <= b_i < len(baseline_world):
                row += [baseline_world[b_i, 0], baseline_world[b_i, 1]]
            else:
                row += ["", ""]
            # MSE（逐点）
            row += [
                mse_probe_map.get(idx, ""),
                mse_base_map.get(idx, "")
            ]
            w.writerow(row)

    print(f"Exported plotting data and point-wise MSE: {fname}")
    print(f"Generated series and MSE plots completed...")
    print()
