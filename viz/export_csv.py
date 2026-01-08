import csv
import numpy as np
from datetime import datetime

from config.runtime import SAMPLE_HZ

def save_csv(app):
    """
    Export various trajectory data into CSV files:
    1) Comprehensive export: ref_pts, sampled, probe_pts, pred_probe_based, baseline
    2) New: Reference trajectory only (preferably evenly timed sampling)

    Args:
        app: The main application instance containing trajectory data.
    """
    print("Exporting trajectory data to CSV...")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_all = f"traj_all_{SAMPLE_HZ}hz_{ts}.csv"
    fname_ref = f"ref_traj_{SAMPLE_HZ}hz_{ts}.csv"  # ✅ Reference trajectory only

    # ---------- Original comprehensive export ----------
    with open(fname_all, "w", newline="") as f:
        w = csv.writer(f)
        # Header
        w.writerow([
            "ref_x", "ref_y",
            "sampled_x", "sampled_y",
            "probe_x", "probe_y",
            "pred_probe_based_x", "pred_probe_based_y",
            "pred_probe_based_varx", "pred_probe_based_vary",
            "baseline_x", "baseline_y",
            "baseline_varx", "baseline_vary"
        ])

        # Various trajectory points (may have different lengths)
        ref = np.asarray(app.ref_pts, dtype=np.float32) if app.ref_pts else np.zeros((0, 2))
        sampled = app.sampled.numpy() if app.sampled is not None else np.zeros((0, 2))
        probe = np.asarray(app.probe_pts, dtype=np.float32) if app.probe_pts else np.zeros((0, 2))

        pred_probe_based = getattr(app, "pred_scaled", None)
        if pred_probe_based is None or len(pred_probe_based) == 0:
            pred_probe_based = np.zeros((0, 2), dtype=np.float64)
        else:
            pred_probe_based = np.asarray(pred_probe_based, dtype=np.float64)

        probe_vars = getattr(app, "pred_vars", None)
        if probe_vars is None or len(probe_vars) == 0:
            probe_vars = np.zeros((0, 2), dtype=np.float64)
        else:
            probe_vars = np.asarray(probe_vars, dtype=np.float64)

        baseline = getattr(app, "baseline_preds", None)
        if baseline is None or len(baseline) == 0:
            baseline = np.zeros((0, 2), dtype=np.float64)
        else:
            baseline = np.asarray(baseline, dtype=np.float64)

        baseline_vars = getattr(app, "baseline_vars", None)
        if baseline_vars is None or len(baseline_vars) == 0:
            baseline_vars = np.zeros((0, 2), dtype=np.float64)
        else:
            baseline_vars = np.asarray(baseline_vars, dtype=np.float64)

        # Find the maximum length and write row by row
        max_len = max(len(ref), len(sampled), len(probe), len(pred_probe_based), len(baseline))
        for i in range(max_len):
            row = []
            # Reference trajectory (original hand-drawn)
            row.extend(ref[i].tolist() if i < len(ref) else ["", ""])
            # Evenly timed sampling (for training)
            row.extend(sampled[i].tolist() if i < len(sampled) else ["", ""])
            # Probe
            row.extend(probe[i].tolist() if i < len(probe) else ["", ""])
            # Probe-based prediction + variance
            if i < len(pred_probe_based):
                row.extend(pred_probe_based[i])
                if i < len(probe_vars):
                    row.extend(probe_vars[i])
                else:
                    row.extend(["", ""])
            else:
                row.extend(["", "", "", ""])
            # Baseline prediction + variance
            if i < len(baseline):
                row.extend(baseline[i])
                if i < len(baseline_vars):
                    row.extend(baseline_vars[i])
                else:
                    row.extend(["", ""])
            else:
                row.extend(["", "", "", ""])

            w.writerow(row)

    # ---------- New: Export reference trajectory only ----------
    # Prefer to export the evenly timed sampled reference trajectory (used for training), fallback to original ref_pts if not available
    if app.sampled is not None and len(app.sampled) > 0:
        ref_for_export = app.sampled.detach().cpu().numpy().astype(np.float64)
        header = ["x", "y"]  # Evenly timed sampling
    else:
        ref_for_export = np.asarray(app.ref_pts, dtype=np.float64) if app.ref_pts else np.zeros((0, 2))
        header = ["x", "y"]  # Original hand-drawn

    with open(fname_ref, "w", newline="") as f2:
        w2 = csv.writer(f2)
        w2.writerow(header)
        for i in range(len(ref_for_export)):
            w2.writerow([ref_for_export[i, 0], ref_for_export[i, 1]])

    print(f"Saved comprehensive data: {fname_all}")
    print(f"Saved reference trajectory only: {fname_ref} (Preferably evenly timed sampling)")
    print("Export completed...")
    print()
