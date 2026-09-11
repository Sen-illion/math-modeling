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
E0_FEB1_KWH = 6000.0
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
SIM_START = OFFICIAL_START

# Adaptive day-level quantile: tune on OFFICIAL_START..TUNE_END, validate on the rest.
TUNE_END = "2025-06-30"
OOS_START = "2025-07-01"
# Nested split inside the tuning window. Parameters are fitted on the inner window and
# the rule is picked on the select window, so the argmax is never taken on the window it
# was scored on and OOS stays untouched during selection.
TUNE_INNER_END = "2025-04-30"
TUNE_SELECT_START = "2025-05-01"
Q_LADDER = (0.6, 0.7, 0.8, 0.85, 0.9)
Q_WARMUP = 0.8
ADAPTIVE_WARMUP_DAYS = 7
# Superseded fixed-margin baseline C_pv7d_q82, kept as the comparison the adaptive gate
# must beat (results/Q2/opt/full_year_summary_C_pv7d_q82.json).
FROZEN_C_Q82_FULL_YEAR_COST = 13765167.58173598
# Official frozen total after adopting the adaptive margin, policy D_pv7d_adaptive
# (results/Q2/opt/full_year_summary.json, results/Q2/result2.xlsx).
FROZEN_OFFICIAL_FULL_YEAR_COST = 13697499.169779435

# Official Q2 V1 keeps no end-of-day SOC value in the day-ahead LP.
TERMINAL_TARGET_KWH = 6000.0
TERMINAL_LAMBDA = 0.0

# V2: leftover-SOC credit in day-ahead / remaining-horizon LP (yuan per kWh).
# This rewards keeping energy, unlike the discarded |E-6000| penalty.
SOC_MU = 0.25
LOAD_QUANTILE = 0.8
PV_QUANTILE = 0.2
PV_SOURCE_V2 = "baseline_7d"
DISPATCH_MODE = "mpc"
MPC_STRIDE = 1
MPC_STRIDE_FALLBACK = 6
PHASE1_MPC_BUDGET_S = 180.0
REMAINING_LP_TIME_LIMIT_S = 8

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
RESULT2_XLSX = REPO_ROOT / "results" / "Q2" / "result2.xlsx"
CLEAN_DIR = REPO_ROOT / "data_clean" / "Q2"
RESULT_DIR = REPO_ROOT / "results" / "Q2"
PHASE1_DIR = RESULT_DIR / "phase1"
FULL_DIR = RESULT_DIR / "full_year"
OPT_DIR = RESULT_DIR / "opt"
ADAPTIVE_DIR = RESULT_DIR / "adaptive"
DIAG_DIR = RESULT_DIR / "diagnostics"
LOG_DIR = RESULT_DIR / "logs"
FORECAST_BANK_NPZ = CLEAN_DIR / "forecast_bank.npz"
ASSET_TABLE_DIR = REPO_ROOT / "paper" / "assets" / "tables"

# Paper Table 1 slots end at HH:10. Table 3 specified dates are from the problem statement.
PAPER_TABLE1_END_MINUTES = (10 * 60 + 10, 12 * 60 + 10, 14 * 60 + 10, 16 * 60 + 10, 18 * 60 + 10, 20 * 60 + 10)
SPECIFIED_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
FOUR_HOUR_BLOCKS = (
    (0, 4 * 60, "0:00-4:00"),
    (4 * 60, 8 * 60, "4:00-8:00"),
    (8 * 60, 12 * 60, "8:00-12:00"),
    (12 * 60, 16 * 60, "12:00-16:00"),
    (16 * 60, 20 * 60, "16:00-20:00"),
    (20 * 60, 24 * 60, "20:00-24:00"),
)
