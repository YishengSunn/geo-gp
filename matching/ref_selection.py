import numpy as np

from config.runtime import ANCHOR_ANGLE, SELECT_HORIZON
from matching.align import get_anchor_correspondence
from utils.misc import closest_index

def choose_best_ref_by_mse(
    refs: list[dict],
    probe_eq_np: np.ndarray,
    probe_raw_np: np.ndarray,
    *,
    horizon: int | None = SELECT_HORIZON,
    align_on_anchor: bool = True
):
    """
    For each reference in self.refs:
    1) Use the original ref/probe to calculate anchors -> get Δθ and scale
    2) Rotate/scale/translate the "reference evenly sampled trajectory" to the probe coordinate system
    3) Calculate the mean squared error (MSE) of (A - B)^2 by corresponding indices
        - If align_on_anchor=True: align the two sequences using the closest index of the anchor point in the evenly sampled sequence before comparison
        - horizon limits the comparison length

    Args:
        refs: list of reference trajectory dicts
        probe_eq_np: (N_pro, 2) np.ndarray of the probe resampled
        probe_raw_np: (N_pro_raw, 2) np.ndarray of the original probe trajectory
        horizon: int | None, maximum number of points to consider for MSE calculation
        align_on_anchor: bool, whether to align the two sequences based on anchor points before MSE calculation

    Returns:
        best_idx: int | None, index of the best matching reference trajectory
        best_pack: tuple | None, (anchor_out, dtheta, scale) of the best matching reference trajectory
        best_mse: float, MSE value of the best matching reference trajectory
    """
    # Note:
    # 使用原始 reference 和 probe 轨迹，分别找到相对于各自起始切线首次跨越 ±ANCHOR_ANGLE 的锚点，
    # 由对应的起点→锚点向量估计两条轨迹之间的旋转角 Δθ 和尺度 scale。
    # 随后可选择是否基于该锚点在重采样后的轨迹上进行对齐，再计算局部重叠段的 MSE。
    print(f"Starting MSE-based reference selection over {len(refs)} candidates...")

    if len(refs) == 0:
        return None, None, float("inf")

    best_idx, best_mse, best_pack = None, float("inf"), None

    for ridx, ref in enumerate(refs):
        ref_raw = ref.get("raw", None)
        if ref_raw is None or ref.get("model_info") is None:
            continue

        # 1) Anchors and scale (using original trajectory)
        out = get_anchor_correspondence(
            ref_raw, probe_raw_np, angle_target=ANCHOR_ANGLE, n_segments_base=10
        )
        v_ref, v_pro = out["ref_vector"], out["probe_vector"]

        dtheta = float(np.arctan2(v_pro[1], v_pro[0]) - np.arctan2(v_ref[1], v_ref[0]))
        scale  = float(np.linalg.norm(v_pro) / max(np.linalg.norm(v_ref), 1e-6))

        # 2) Rotate/scale/translate the "reference evenly sampled trajectory" to the probe coordinate system
        ref_samp = ref["sampled"].detach().cpu().numpy()  # (Nr, 2)
        c, s = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s], [s,  c]], dtype=np.float64)
        ref_in_probe = (ref_samp - ref_samp[0]) @ R.T * scale + probe_eq_np[0]

        # 3) Select the aligned overlapping segment and calculate MSE
        if align_on_anchor:
            i_ref_res = closest_index(out["ref_point"], ref_samp)
            i_pro_res = closest_index(out["probe_point"], probe_eq_np)
            offset = int(i_pro_res - i_ref_res)

            start_ref = max(0, -offset)
            start_pro = max(0, offset)
            n_overlap = min(ref_in_probe.shape[0] - start_ref, probe_eq_np.shape[0] - start_pro)  # Overlapping length
            if n_overlap <= 0:
                continue
            if horizon is not None:
                n_overlap = min(n_overlap, int(horizon))

            A = ref_in_probe[start_ref : start_ref + n_overlap]
            B = probe_eq_np[start_pro : start_pro + n_overlap]
        else:
            n_overlap = min(ref_in_probe.shape[0], probe_eq_np.shape[0])
            if n_overlap <= 0:
                continue
            if horizon is not None:
                n_overlap = min(n_overlap, int(horizon))
            A = ref_in_probe[:n_overlap]
            B = probe_eq_np[:n_overlap]

        mse = float(np.mean(np.sum((A - B) ** 2, axis=1)))

        if mse < best_mse:
            best_mse  = mse
            best_idx  = ridx
            best_pack = (out, dtheta, scale)

    print(f"MSE-based selection completed: best_idx={best_idx}, best_mse={best_mse:.6f}, "
          f"Δθ={np.degrees(best_pack[1]) if best_pack else None:.2f}°, scale={best_pack[2] if best_pack else None:.3f}")
    print()

    return best_idx, best_pack, best_mse
