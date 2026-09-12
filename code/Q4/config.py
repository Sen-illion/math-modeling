"""Q4 paths. Do not write result2.xlsx / result3.xlsx."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
N_INTERVALS = 144
N_DAYS_YEAR = 365
N_OFFICIAL_DAYS = 334
OFFICIAL_START = "2025-02-01"
OFFICIAL_END = "2025-12-31"
PHASE1_END = "2025-02-14"
N_PHASE1_DAYS = 14
TUNE_END = "2025-06-30"
OOS_START = "2025-07-01"
TUNE_INNER_END = "2025-04-30"
TUNE_SELECT_START = "2025-05-01"
CURRENT_Q42_QMIN = 0.65
CURRENT_Q42_K = 0.20
CURRENT_Q43_BETA_LOCK = 1.0
CURRENT_Q43_BETA_OPEN = 0.2
SPECIFIED_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
RNG_SEED = 42
E0_FEB1_KWH = 6000.0
ABS_TOL_KWH = 1e-3

ATTACHMENT1_XLSX = REPO_ROOT / "data_raw" / "Q3" / "attachment1.xlsx"
ATTACHMENT4_CANDIDATES = (
    REPO_ROOT / "data_raw" / "Q4" / "attachment4.xlsx",
    REPO_ROOT / "problem_files" / "附件" / "附件4.xlsx",
)
RESULT4_2_TEMPLATE = REPO_ROOT / "problem_files" / "附件" / "附件5" / "result4-2.xlsx"
RESULT4_3_TEMPLATE = REPO_ROOT / "problem_files" / "附件" / "附件5" / "result4-3.xlsx"

RESULT_DIR = REPO_ROOT / "results" / "Q4"
RESULT4_2_XLSX = RESULT_DIR / "result4-2.xlsx"
RESULT4_3_XLSX = RESULT_DIR / "result4-3.xlsx"
PHASE1_DIR = RESULT_DIR / "phase1"
HAT0_DIAG_DIR = RESULT_DIR / "diagnostics" / "hat0_expanding_layers"
DIAG_DIR = RESULT_DIR / "diagnostics" / "intraday_price_correction"
ORACLE_DIAG_DIR = RESULT_DIR / "diagnostics" / "oracle_price"
ALIGN_DIAG_DIR = RESULT_DIR / "diagnostics" / "align_current_policies"
RETUNE_DIAG_DIR = RESULT_DIR / "diagnostics" / "retune_q_beta"
ALIGN_QUANTILE_DIAG_DIR = RESULT_DIR / "diagnostics" / "align_q3_quantile"
ALIGN_OFFICIAL_DIAG_DIR = RESULT_DIR / "diagnostics" / "align_q2q3_official"
Q2_SELECTED_RULE = REPO_ROOT / "results" / "Q2" / "adaptive" / "selected_rule.json"

Q2_POLICY = {
    "name": "C_pv7d_q82",
    "pv_source": "baseline_7d",
    "q_load": 0.8,
    "q_pv": 0.2,
    "soc_mu": 0.0,
    "dispatch": "greedy",
}
Q2_ADAPTIVE_POLICY = {
    "name": "D_pv7d_adaptive",
    "pv_source": "baseline_7d",
    "soc_mu": 0.0,
    "dispatch": "greedy",
}
Q2_NETRHO_POLICY = {
    "name": "E_pv7d_netrho",
    "pv_source": "baseline_7d",
    "soc_mu": 0.0,
    "dispatch": "rho",
}
Q3_POLICY = "LA"

FOUR_HOUR_BLOCKS = (
    (0, 4 * 60, "0:00-4:00"),
    (4 * 60, 8 * 60, "4:00-8:00"),
    (8 * 60, 12 * 60, "8:00-12:00"),
    (12 * 60, 16 * 60, "12:00-16:00"),
    (16 * 60, 20 * 60, "16:00-20:00"),
    (20 * 60, 24 * 60, "20:00-24:00"),
)
PAPER_TABLE1_END_MINUTES = (
    10 * 60 + 10,
    12 * 60 + 10,
    14 * 60 + 10,
    16 * 60 + 10,
    18 * 60 + 10,
    20 * 60 + 10,
)
