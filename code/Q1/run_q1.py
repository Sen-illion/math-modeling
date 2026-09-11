"""Q1 entry point: clean data, solve LP, validate, export tables and figures."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

Q1_DIR = Path(__file__).resolve().parent
CODE_DIR = Q1_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(Q1_DIR) not in sys.path:
    sys.path.insert(0, str(Q1_DIR))

from baselines import baseline_no_storage, baseline_tou_greedy
from config import (
    ATTACHMENT1_XLSX,
    LOG_DIR,
    RESULT1_TEMPLATE_XLSX,
    RESULT_DIR,
)
from export_results import export_paper_tables, export_result1
from load_data import load_intervals, write_clean_tables
from model_lp import solve_lp
from plotting import plot_all
from validate import summarize, validate_dispatch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "solve.log"
    started = time.perf_counter()

    frame = load_intervals(ATTACHMENT1_XLSX)
    profile = write_clean_tables(frame, ATTACHMENT1_XLSX)

    b0 = baseline_no_storage(frame)
    b1 = baseline_tou_greedy(frame)
    lp = solve_lp(frame)
    lp["cost"] = float(np.dot(frame["price"].to_numpy(), lp["purchase_kwh"]))
    b0["cost"] = float(np.dot(frame["price"].to_numpy(), b0["purchase_kwh"]))
    b1["cost"] = float(np.dot(frame["price"].to_numpy(), b1["purchase_kwh"]))

    metrics = {
        "B0": summarize(frame, b0, "B0"),
        "B1": summarize(frame, b1, "B1"),
        "M0": summarize(frame, lp, "M0"),
    }

    errors = validate_dispatch(frame, lp, require_cost_below_b0=metrics["B0"]["cost"])
    elapsed = time.perf_counter() - started
    log_path.write_text(
        "\n".join(
            [
                f"status={lp['status']}",
                f"objective={lp['objective']:.6f}",
                f"elapsed_s={elapsed:.3f}",
                f"errors={len(errors)}",
                *errors,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if errors:
        (RESULT_DIR / "metrics.json").write_text(
            json.dumps({"metrics": metrics, "validation_errors": errors}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise SystemExit("Q1 validation failed:\n" + "\n".join(errors))

    result_xlsx = export_result1(frame, lp)
    table1_path, table2_path = export_paper_tables(frame, lp, metrics)
    figure_paths = plot_all(frame, lp, metrics)

    payload = {
        "question": "Q1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "solver": lp["solver"],
        "status": lp["status"],
        "elapsed_s": elapsed,
        "power_limit": "microgrid_interface_5000kW",
        "inputs": {
            "attachment1": str(ATTACHMENT1_XLSX),
            "attachment1_sha256": _sha256(ATTACHMENT1_XLSX),
            "template": str(RESULT1_TEMPLATE_XLSX),
            "template_sha256": _sha256(RESULT1_TEMPLATE_XLSX),
        },
        "profile": profile,
        "metrics": metrics,
        "outputs": {
            "result1_xlsx": str(result_xlsx),
            "table1_csv": str(table1_path),
            "table2_csv": str(table2_path),
            "figures": [str(path) for path in figure_paths],
        },
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT_DIR / "run_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"wrote {result_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
