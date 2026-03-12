import os
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

PIPELINE = REPO_ROOT / "CHIP_mtr_data" / "CHIP_mtr_preprocess.py"
MONITOR_1 = REPO_ROOT / "CHIP_mtr_1" / "CHIP_mtr1_ai_stability_drift.py"
MONITOR_2 = REPO_ROOT / "CHIP_mtr_2" / "CHIP_mtr2_performance.py"
MONITOR_3 = REPO_ROOT / "CHIP_mtr_3" / "CHIP_mtr3_hitl_stability.py"
M1_RESULTS = REPO_ROOT / "CHIP_mtr_1" / "CHIP_mtr_1_test_results.json"
M2_RESULTS = REPO_ROOT / "CHIP_mtr_2" / "CHIP_mtr_2_test_results.json"
M3_RESULTS = REPO_ROOT / "CHIP_mtr_3" / "CHIP_mtr_3_test_results.json"
CHIP_MASTER = REPO_ROOT / "CHIP_mtr_data" / "CHIP_data" / "CHIP_master.csv"
CHIP_BASELINE = REPO_ROOT / "CHIP_mtr_data" / "CHIP_data" / "CHIP_baseline.csv"
CHIP_COMPARATOR = REPO_ROOT / "CHIP_mtr_data" / "CHIP_data" / "CHIP_comparator.csv"
ANALYSIS_REPORT = REPO_ROOT / "docs" / "CHIP_mtr_test_results_analysis.md"


def _build_env() -> dict:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    repo_root_str = str(REPO_ROOT)
    env["PYTHONPATH"] = (
        f"{repo_root_str}{os.pathsep}{current_pythonpath}"
        if current_pythonpath
        else repo_root_str
    )
    return env


def _run_step(name: str, script_path: Path, env: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"Running: {script_path}")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    print(f"{name} completed.")


def _count_csv_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8") as handle:
        row_count = sum(1 for _ in handle)
    return max(row_count - 1, 0)


def _load_results(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list) and payload:
        return payload[0] if isinstance(payload[0], dict) else {}
    return payload if isinstance(payload, dict) else {}


def _metric_value(summary_rows: list, metric_name: str):
    for row in summary_rows:
        if str(row.get("Metric", "")).strip().lower() == metric_name.strip().lower():
            return row.get("Value")
    return None


def _fmt_number(value, digits: int = 4) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return str(value)


def _fmt_percent(value) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return "n/a"


def _status_for_threshold(value, threshold: float, greater_is_better: bool) -> str:
    if not isinstance(value, (int, float)):
        return "Needs review"
    if greater_is_better:
        return "Met" if value >= threshold else "Needs review"
    return "Met" if value <= threshold else "Needs review"


def _to_number(value, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _build_pie_mermaid(title: str, categories: list, counts: list) -> str:
    pairs = []
    for idx, category in enumerate(categories or []):
        count_value = counts[idx] if idx < len(counts or []) else 0
        pairs.append((str(category), _to_number(count_value, 0.0)))
    if not pairs:
        pairs = [("No Data", 1.0)]

    lines = ["```mermaid", "pie showData", f'    title "{title}"']
    for label, value in pairs:
        lines.append(f'    "{label}" : {value}')
    lines.append("```")
    return "\n".join(lines)


def _format_run_timestamps() -> str:
    """
    Return dual time representation for NJ (ET) and Hyderabad (IST).
    Uses fixed UTC offsets requested by stakeholders.
    """
    utc_now = datetime.now(timezone.utc)
    et_tz = timezone(timedelta(hours=-4))
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    et_now = utc_now.astimezone(et_tz)
    ist_now = utc_now.astimezone(ist_tz)
    et_str = et_now.strftime("%Y-%m-%d %H:%M:%S")
    ist_str = ist_now.strftime("%Y-%m-%d %H:%M:%S")
    return f"{et_str} ET (UTC-4) / {ist_str} IST (UTC+5.5)"


def _build_analysis_markdown() -> str:
    m1 = _load_results(M1_RESULTS)
    m2 = _load_results(M2_RESULTS)
    m3 = _load_results(M3_RESULTS)

    m1_summary = m1.get("ai_stability_drift_summary_table") or []
    m2_summary = m2.get("ai_hitl_concordance_summary_table") or []
    m3_summary = m3.get("hitl_calibration_summary_table") or []

    master_count = _count_csv_rows(CHIP_MASTER)
    baseline_count = _count_csv_rows(CHIP_BASELINE)
    comparator_count = _count_csv_rows(CHIP_COMPARATOR)

    m1_psi = _metric_value(m1_summary, "Overall Prediction Shift (PSI)")
    m2_accuracy = _metric_value(m2_summary, "Accuracy")
    m2_precision = _metric_value(m2_summary, "Precision")
    m2_recall = _metric_value(m2_summary, "Recall")
    m2_f1 = _metric_value(m2_summary, "F1 Score")
    m2_auc = _metric_value(m2_summary, "Auc")
    m3_team_reject = _metric_value(m3_summary, "Team Avg Rejection Rate")

    m1_status = _status_for_threshold(m1_psi, threshold=0.1, greater_is_better=False)
    m2_status = _status_for_threshold(m2_accuracy, threshold=0.95, greater_is_better=True)

    run_timestamp_dual = _format_run_timestamps()

    m1_pie = m1.get("ai_outcome_mix_pie_chart") or {}
    m2_pie = m2.get("hitl_class_balance_pie_chart") or {}
    m3_pie = m3.get("hitl_decision_mix_pie_chart") or {}
    m3_timeline = m3.get("hitl_rejection_volume_time_line_graph", {}).get("data", {})
    m3_daily_rejection = m3_timeline.get("daily_rejection_rate") or []
    m3_daily_volume = m3_timeline.get("daily_review_volume") or []
    m3_last_date = m3_daily_rejection[-1][0] if m3_daily_rejection else "n/a"
    m3_last_rejection = m3_daily_rejection[-1][1] if m3_daily_rejection else None
    m3_last_volume = m3_daily_volume[-1][1] if m3_daily_volume else None

    m1_pie_block = _build_pie_mermaid(
        "MTR 1 - Comparator AI Outcome Mix",
        m1_pie.get("categories") or [],
        (m1_pie.get("data") or {}).get("outcome_count") or [],
    )
    m2_pie_block = _build_pie_mermaid(
        "MTR 2 - Comparator HITL Class Balance",
        m2_pie.get("categories") or [],
        (m2_pie.get("data") or {}).get("decision_count") or [],
    )
    m3_pie_block = _build_pie_mermaid(
        "MTR 3 - Comparator HITL Decision Mix",
        m3_pie.get("categories") or [],
        (m3_pie.get("data") or {}).get("decision_count") or [],
    )

    return f"""# CHIP Monitor Test Results – Auto-Generated Run Summary

**Run:** CHIP_mtr_data (preprocessing) -> MTR 1 -> MTR 2 -> MTR 3  
**Execution command:** `python run_all_monitors.py`  
**Generated at:** {run_timestamp_dual}

---

## 1. Latest Run Summary

| Step | Description | Result |
|---|---|---|
| **CHIP_mtr_data** | ETL preprocessing monitor | OK - master={master_count}, baseline={baseline_count}, comparator={comparator_count} |
| **MTR 1** | Model Output Stability (Drift) | OK - PSI (ai_overall_status) = {_fmt_number(m1_psi, 6)} |
| **MTR 2** | Approval Concordance | OK - Accuracy = {_fmt_percent(m2_accuracy)}, Precision = {_fmt_number(m2_precision)}, Recall = {_fmt_number(m2_recall)}, F1 = {_fmt_number(m2_f1)}, AUC = {_fmt_number(m2_auc)} |
| **MTR 3** | QA Calibration | OK - Team Avg Rejection Rate = {_fmt_percent(m3_team_reject)} |

---

## 2. Proposal Criteria Snapshot

| Monitor | Criterion | This run | Status |
|---|---|---|---|
| **MTR 1** | Reliability target: `PSI < 0.1` | PSI = {_fmt_number(m1_psi, 6)} | **{m1_status}** |
| **MTR 2** | Agreement target: `Accuracy > 95%` | Accuracy = {_fmt_percent(m2_accuracy)} | **{m2_status}** |
| **MTR 3** | Stable QA rejection behavior | Team Avg Rejection Rate = {_fmt_percent(m3_team_reject)} | **Observed** |

---

## 3. Visual Summary (GitHub Mermaid Compatible)

```mermaid
flowchart LR
    etl["CHIP_mtr_data ETL<br/>master={master_count}<br/>baseline={baseline_count}<br/>comparator={comparator_count}"] --> m1["MTR 1<br/>PSI={_fmt_number(m1_psi, 6)}<br/>Status={m1_status}"]
    etl --> m2["MTR 2<br/>Accuracy={_fmt_percent(m2_accuracy)}<br/>AUC={_fmt_number(m2_auc)}<br/>Status={m2_status}"]
    etl --> m3["MTR 3<br/>Team Reject Rate={_fmt_percent(m3_team_reject)}<br/>Observed Date={m3_last_date}"]
```

{m1_pie_block}

{m2_pie_block}

{m3_pie_block}

```mermaid
flowchart LR
    d["Latest MTR 3 Day<br/>{m3_last_date}"] --> rr["Daily Rejection Rate<br/>{_fmt_percent(m3_last_rejection)}"]
    d --> rv["Daily Review Volume<br/>{_fmt_number(m3_last_volume, 0)}"]
```

---

## 4. Key Output Files

- `CHIP_mtr_data/CHIP_data/CHIP_master.csv`
- `CHIP_mtr_data/CHIP_data/CHIP_baseline.csv`
- `CHIP_mtr_data/CHIP_data/CHIP_comparator.csv`
- `CHIP_mtr_1/CHIP_mtr_1_test_results.json`
- `CHIP_mtr_2/CHIP_mtr_2_test_results.json`
- `CHIP_mtr_3/CHIP_mtr_3_test_results.json`

---

## 5. Notes

- This report is regenerated automatically at the end of each successful chain run.
- If AUC shows `null`, comparator labels are typically single-class for the current run window.
- Canonical troubleshooting decoder: see `../README.md` -> **Master Troubleshooting Table (Canonical)**.
"""


def _write_analysis_report() -> None:
    report_md = _build_analysis_markdown()
    ANALYSIS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_REPORT.write_text(report_md, encoding="utf-8")
    print(f"Analysis report updated: {ANALYSIS_REPORT}")


def main() -> int:
    env = _build_env()
    steps = [
        ("Preprocessing", PIPELINE),
        ("Monitor 1", MONITOR_1),
        ("Monitor 2", MONITOR_2),
        ("Monitor 3", MONITOR_3),
    ]

    for name, script_path in steps:
        if not script_path.exists():
            print(f"[ERROR] Missing script: {script_path}")
            return 1
        try:
            _run_step(name, script_path, env)
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] {name} failed with exit code {exc.returncode}.")
            return exc.returncode

    try:
        print("\n=== Analysis Report ===")
        _write_analysis_report()
    except Exception as exc:
        print(f"[ERROR] Failed to update analysis report: {exc}")
        return 1

    print("\nAll steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
