"""Q2 frozen parameters. Efficiency one-way 0.9 is a modeling interpretation."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DT_HOURS = 10.0 / 60.0
N_INTERVALS = 144
N_DAYS_YEAR = 365

E_CAPACITY_KWH = 12000.0
E_MIN_KWH = 1200.0
E_MAX_KWH = 10800.0
E0_JAN1_KWH = 6000.0
P_MAX_KW = 5000.0
ETA_CHARGE = 0.9
ETA_DISCHARGE = 0.9
P_MAX_KWH = P_MAX_KW * DT_HOURS  # 833.333... kWh on the AC interface

ABS_TOL_KWH = 1e-3
REL_TOL = 1e-4
SIMULTANEOUS_TOL = 1e-4
SOLVER_TIME_LIMIT_S = 60
RNG_SEED = 42

OFFICIAL_START = "2025-02-01"
OFFICIAL_END = "2025-12-31"
PHASE1_END = "2025-02-14"

# Official Q2 keeps no end-of-day SOC value in the day-ahead LP.
TERMINAL_TARGET_KWH = 6000.0
TERMINAL_LAMBDA = 0.0

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 4,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": 4,
    "random_state": RNG_SEED,
    "verbosity": 0,
}

FEATURE_COLS = [
    "slot",
    "dow",
    "month",
    "doy",
    "lag_1d",
    "lag_2d",
    "lag_7d",
    "mean_3d",
    "mean_7d",
    "std_7d",
    "mean_30d",
]

MODEL_NAMES = ("baseline_7d", "xgb_expanding", "xgb_rolling30", "xgb_rolling60")

ATTACHMENT1_XLSX = REPO_ROOT / "data_raw" / "Q2" / "attachment1.xlsx"
ATTACHMENT2_XLSX = REPO_ROOT / "data_raw" / "Q2" / "attachment2.xlsx"
RESULT2_TEMPLATE_XLSX = REPO_ROOT / "data_raw" / "Q2" / "result2_template.xlsx"
CLEAN_DIR = REPO_ROOT / "data_clean" / "Q2"
RESULT_DIR = REPO_ROOT / "results" / "Q2"
PHASE1_DIR = RESULT_DIR / "phase1"
FULL_DIR = RESULT_DIR / "full_year"
LOG_DIR = RESULT_DIR / "logs"
