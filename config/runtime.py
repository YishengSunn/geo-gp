from pathlib import Path
from .config import Config


# ============================================================
# Load YAML config
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config" / "default.yaml"

CFG = Config(CFG_PATH)

# ============================================================
# Random seed
# ============================================================

SEED = CFG.experiment["seed"]

# ============================================================
# Sampling
# ============================================================

K_HIST = CFG.sampling["k_hist"]                # Seed history length
SAMPLE_HZ = CFG.sampling["sample_hz"]          # Sample frequency for equal-dt resampling
DEFAULT_SPEED = CFG.sampling["default_speed"]  # Convert polyline length to time for equal-time sampling
SMOOTH_WIN = CFG.sampling["smooth_win"]        # Smoothing window size for trajectory smoothing

# ============================================================
# Dataset parameters
# ============================================================

TRAIN_RATIO = CFG.dataset["train_ratio"]       # Training ratio (1.0 means full training)

# ============================================================
# Core GP parameters
# ============================================================

NEAREST_K = CFG.gp["nearest_k"]
MAX_EXPERTS = CFG.gp["max_experts"]
MAX_DATA_PER_EXPERT = CFG.gp["max_data_per_expert"]
MIN_POINTS_OFFLINE = CFG.gp["min_points_offline"]
WINDOW_SIZE = CFG.gp["window_size"]            # Sliding window size (None means not used)

# ============================================================
# Method settings
# ============================================================

METHOD_ID = CFG.method["id"]
METHOD_HPARAM = CFG.method["hparams"]          # Hyperparameters for the method

# ============================================================
# Prediction
# ============================================================

ROLLOUT_HORIZON = CFG.prediction["rollout_horizon"]  # Max steps to rollout
MSE_THRESH = CFG.prediction["mse_thresh"]            # Threshold for geometric drift detection
GOAL_STOP_EPS = CFG.prediction["goal_stop_eps"]      # Stop if within this distance to goal
MAX_START_JUMP = CFG.prediction["max_start_jump"]    # Max allowed jump at the start of the trajectory
DROP_K = CFG.prediction["drop_k"]                    # Number of probe points to drop on each retry after drift detection
MAX_RETRIES = CFG.prediction["max_retries"]          # Max number of retries for prediction after drift detection
