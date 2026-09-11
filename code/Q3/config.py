"""Q3 frozen parameters. AC-side efficiency matches Q2."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DT_HOURS = 10.0 / 60.0
N_INTERVALS = 144
N_DAYS_YEAR = 365

E_CAPACITY_KWH = 12000.0
E_MIN_KWH = 1200.0
E_MAX_KWH = 10800.0
E0_JAN1_KWH = 6000.0
E_REF_KWH = 6000.0
P_MAX_KW = 5000.0
ETA_CHARGE = 0.9
ETA_DISCHARGE = 0.9
P_MAX_KWH = P_MAX_KW * DT_HOURS

RNG_SEED = 42
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

# Previous frozen official total (M1, zero p0); only replace result3 if the new winner is cheaper.
PREVIOUS_OFFICIAL_M1_COST = 13519092.792762008
BETA_LOCK = 1.0
BETA_OPEN = 0.2
LOCK_SLOTS = 36
TERMINAL_LAMBDA = 0.0
SELECT_EPS = 0.03
ADJ_PV_L1_KWH = 3000.0
SIGMA_MIN_SAMPLES = 3

# 插值起点口径。"measured" 是 2026-09-11 冻结值：取发布前最后一个已完成时段的实测光伏，
# 代替把 6:00/18:00 的起点硬设为 0。"zero" 保留旧口径，只用于消融。
PV_P0_MODE = "measured"
PV_P0_MODES = ("zero", "measured")
BETA_LOCK_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
BETA_OPEN_GRID = (0.0, 0.2, 0.5, 1.0)

# Playback SOC reserve. 0.0 keeps the price-blind greedy rule shared with Q2.
RESERVE_GAMMA = 0.0

ABS_TOL_KWH = 1e-3
REL_TOL = 1e-4
SIMULTANEOUS_TOL = 1e-4
SOLVER_TIME_LIMIT_S = 60

OFFICIAL_START = "2025-02-01"
OFFICIAL_END = "2025-12-31"
PHASE1_END = "2025-02-14"
SPECIFIED_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")

ISSUE_HOURS = (0, 6, 12, 18)
ISSUE_P0_ZERO = {0, 6, 18}

ATTACHMENT1_XLSX = REPO_ROOT / "data_raw" / "Q3" / "attachment1.xlsx"
ATTACHMENT2_XLSX = REPO_ROOT / "data_raw" / "Q3" / "attachment2.xlsx"
ATTACHMENT3_XLSX = REPO_ROOT / "data_raw" / "Q3" / "attachment3.xlsx"
RESULT3_TEMPLATE_XLSX = REPO_ROOT / "data_raw" / "Q3" / "result3_template.xlsx"
CLEAN_DIR = REPO_ROOT / "data_clean" / "Q3"
RESULT_DIR = REPO_ROOT / "results" / "Q3"
PHASE1_DIR = RESULT_DIR / "phase1"
EXP_DIR = RESULT_DIR / "exp"
LOG_DIR = RESULT_DIR / "logs"
ASSET_TABLE_DIR = REPO_ROOT / "paper" / "assets" / "tables"

PAPER_TABLE1_END_MINUTES = (
    10 * 60 + 10,
    12 * 60 + 10,
    14 * 60 + 10,
    16 * 60 + 10,
    18 * 60 + 10,
    20 * 60 + 10,
)

FOUR_HOUR_BLOCKS = (
    (0, 4 * 60, "0:00-4:00"),
    (4 * 60, 8 * 60, "4:00-8:00"),
    (8 * 60, 12 * 60, "8:00-12:00"),
    (12 * 60, 16 * 60, "12:00-16:00"),
    (16 * 60, 20 * 60, "16:00-20:00"),
    (20 * 60, 24 * 60, "20:00-24:00"),
)
