import math
import numpy as np
from pathlib import Path
from .config import Config


# ============================================================
# Load YAML config
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config" / "default.yaml"

CFG = Config(CFG_PATH)

# ============================================================
# Experiment / random seed
# ============================================================

SEED = CFG.experiment["seed"]

# ============================================================
# Dataset parameters
# ============================================================

TRAIN_RATIO = CFG.dataset["train_ratio"]       # Training ratio (1.0 means full training)

# ============================================================
# Sampling / time
# ============================================================

SAMPLE_HZ = CFG.sampling["sample_hz"]          # Sample frequency for equal-dt resampling
DEFAULT_SPEED = CFG.sampling["default_speed"]  # Convert polyline length to time for equal-time sampling

# ============================================================
# Core GP / rollout parameters
# ============================================================

K_HIST = CFG.sampling["k_hist"]      # Seed history length
NEAREST_K = CFG.gp["nearest_k"]
MAX_EXPERTS = CFG.gp["max_experts"]
MAX_DATA_PER_EXPERT = CFG.gp["max_data_per_expert"]
MIN_POINTS_OFFLINE = CFG.gp["min_points_offline"]
WINDOW_SIZE = CFG.gp["window_size"]  # Sliding window size (None means not used)

# ============================================================
# Method settings
# ============================================================

METHOD_ID = CFG.method["id"]            # 1: polar->delta, 5: polar+delta->delta
METHOD_CONFIGS = CFG.method["configs"]  # List of method configurations
METHOD_HPARAM = CFG.method["hparams"]   # Hyperparameters for the method
MATCH_MODE = CFG.matching["mode"]       # Can switch between similarity / affine / angle (press M key)

# ============================================================
# Geometry / anchor related
# ============================================================

MIN_START_ANGLE_DIFF_DEG = CFG.anchor["min_start_angle_diff_deg"]  # Minimum start angle difference
MIN_START_ANGLE_DIFF = math.radians(MIN_START_ANGLE_DIFF_DEG)
ANCHOR_ANGLE = np.radians(CFG.anchor['anchor_angle_deg'])          # Anchor angle based on relative start tangent
PHI0_K_REF = CFG.anchor['phi0_k_ref']                              # Number of initial points used to estimate phi0 of the reference trajactory (used for getting anchor correspondence)
PHI0_K_PROBE = CFG.anchor['phi0_k_probe']                          # Number of initial points used to estimate phi0 of the probe trajectory (used for getting anchor correspondence)
SELECT_HORIZON = CFG.anchor['select_horizon']                      # Only compare the first few overlapping points' MSE when selecting the best reference

# ============================================================
# UI settings
# ============================================================

DOMAIN = CFG.ui["domain"]
LINE_WIDTHS = CFG.ui["line_widths"]
