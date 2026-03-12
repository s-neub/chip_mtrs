"""
CHIP_mtr_data: ModelOp Center ETL / Preprocessing Monitor
---------------------------------------------------------
Consumes raw business data (activity log, feedback, AI responses) and produces
master, baseline, and comparator CSV datasets for downstream CHIP_mtr_1, CHIP_mtr_2, CHIP_mtr_3.
Set input assets in the Add monitor wizard; outputs become implementation assets
for the stability, performance, and HITL monitors.
"""

import os
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
_JOB_PARAMETERS_PATH = os.path.join(_SCRIPT_DIR, "job_parameters.json")

from CHIP_mtr_preprocess import execute_pipeline_csv_only

JOB = {}
JOB_PARAMETERS = {}


def _load_local_job_parameters() -> dict:
    if not os.path.exists(_JOB_PARAMETERS_PATH):
        return {}
    try:
        with open(_JOB_PARAMETERS_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


LOCAL_JOB_PARAMETERS = _load_local_job_parameters()

# modelop.init
def init(job_json: dict) -> None:
    """Store and normalize job parameters once, following ModelOp init pattern."""
    global JOB, JOB_PARAMETERS
    JOB = job_json or {}
    raw_payload = JOB.get("rawJson") if isinstance(JOB, dict) else None

    parsed = {}
    try:
        if isinstance(raw_payload, str):
            parsed = json.loads(raw_payload)
        elif isinstance(raw_payload, dict):
            parsed = raw_payload
    except Exception:
        parsed = {}

    runtime_params = parsed.get("jobParameters") if isinstance(parsed, dict) else {}
    runtime_params = runtime_params if isinstance(runtime_params, dict) else {}
    JOB_PARAMETERS = {**LOCAL_JOB_PARAMETERS, **runtime_params}


# modelop.metrics
def metrics() -> dict:
    """
    Run the ETL pipeline and write CHIP_master.csv, CHIP_baseline.csv, CHIP_comparator.csv.
    Input paths (activity_file, feedback_file, ai_responses_dir) and output_dir are read from
    job_json jobParameters or from implementation assets. Output CSVs are written to output_dir
    (platform may capture this as implementation assets for downstream monitors).
    """
    params = JOB_PARAMETERS or LOCAL_JOB_PARAMETERS
    output_dir = params.get("output_dir") or os.environ.get("CHIP_ETL_OUTPUT_DIR", os.path.join(_SCRIPT_DIR, "CHIP_data"))
    activity_file = params.get("activity_file")
    feedback_file = params.get("feedback_file")
    ai_responses_dir = params.get("ai_responses_dir")
    split_method = params.get("split_method", "DATE")
    days_threshold = params.get("days_threshold")
    volume_threshold = int(params.get("volume_threshold", 5000))
    baseline_start_date = params.get("baseline_start_date")
    min_records_baseline = int(params.get("min_records_baseline", 20))
    min_records_comparator = int(params.get("min_records_comparator", 20))

    result = execute_pipeline_csv_only(
        output_dir=output_dir,
        split_method=split_method,
        days_threshold=days_threshold,
        volume_threshold=volume_threshold,
        baseline_start_date=baseline_start_date,
        activity_file=activity_file,
        feedback_file=feedback_file,
        ai_responses_dir=ai_responses_dir,
        min_records_baseline=min_records_baseline,
        min_records_comparator=min_records_comparator,
        job_parameters=params,
    )
    yield result


if __name__ == "__main__":
    # Local run: write CSVs to CHIP_mtr_data/CHIP_data for downstream monitors
    output_dir = os.path.join(_SCRIPT_DIR, "CHIP_data")
    mock_job = {
        "rawJson": json.dumps({
            "jobParameters": {
                "output_dir": output_dir
            }
        })
    }
    init(mock_job)
    results = list(metrics())
    print("CHIP_mtr_data monitor output:", results[0] if results else {})
