import numpy as np

from geometry.resample import resample_by_arclen_fraction


def geom_mse(hist: np.ndarray, ref: np.ndarray, M: int = 80) -> float:
    """
    Compute geometric MSE between two trajectories by resampling them to M points
    equally spaced by arc-length fraction.

    Args:
        hist: (N, D) array of historical trajectory points
        ref: (N, D) array of reference trajectory points
        M: number of points to resample to

    Returns:
        Geometric MSE value    
    """
    H = resample_by_arclen_fraction(hist, M)
    R = resample_by_arclen_fraction(ref, M)
    return float(np.mean((H - R) ** 2))
