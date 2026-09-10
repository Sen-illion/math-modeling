"""Frozen Q1 parameters from Appendix 1 and the method plan."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DT_HOURS = 1.0 / 6.0
N_INTERVALS = 144
E_CAPACITY_KWH = 12000.0
E_MIN_KWH = 1200.0
E_MAX_KWH = 10800.0
E0_KWH = 6000.0
P_MAX_KW = 5000.0
ETA = 0.9
P_MAX_KWH = P_MAX_KW * DT_HOURS

ABS_TOL_KWH = 1e-3
REL_TOL = 1e-4
SOLVER_TIME_LIMIT_S = 60

ATTACHMENT1_XLSX = REPO_ROOT / "data_raw" / "Q1" / "attachment1.xlsx"
RESULT1_TEMPLATE_XLSX = REPO_ROOT / "data_raw" / "Q1" / "result1_template.xlsx"
CLEAN_DIR = REPO_ROOT / "data_clean" / "Q1"
RESULT_DIR = REPO_ROOT / "results" / "Q1"
FIGURE_DIR = RESULT_DIR / "figures"
LOG_DIR = RESULT_DIR / "logs"
ASSET_FIGURE_DIR = REPO_ROOT / "paper" / "assets" / "figures"
ASSET_TABLE_DIR = REPO_ROOT / "paper" / "assets" / "tables"

# Paper Table 1 intervals are clock-aligned 10-minute slots; take the slot ending at HH:10.
PAPER_TABLE1_END_MINUTES = (10 * 60 + 10, 12 * 60 + 10, 14 * 60 + 10, 16 * 60 + 10, 18 * 60 + 10, 20 * 60 + 10)

FOUR_HOUR_BLOCKS = (
    (0, 4 * 60, "0:00-4:00"),
    (4 * 60, 8 * 60, "4:00-8:00"),
    (8 * 60, 12 * 60, "8:00-12:00"),
    (12 * 60, 16 * 60, "12:00-16:00"),
    (16 * 60, 20 * 60, "16:00-20:00"),
    (20 * 60, 24 * 60, "20:00-24:00"),
)
