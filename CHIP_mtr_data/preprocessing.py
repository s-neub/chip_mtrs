"""
CHIP_mtr_data: ModelOp Center ETL / Preprocessing Monitor
---------------------------------------------------------
Consumes raw business data (activity log, feedback, AI responses) and produces
master, baseline, and comparator CSV datasets for downstream CHIP_mtr_1, CHIP_mtr_2, CHIP_mtr_3.
Configure input assets in the Add monitor wizard; outputs become implementation assets
for the stability, performance, and HITL monitors.
"""

import os
import json

# Allow importing the pipeline from the parent repo when this monitor lives under chip_mtrs
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
if _PARENT not in __import__('sys').path:
    __import__('sys').path.insert(0, _PARENT)

from CHIP_mtr_preprocess import execute_pipeline_csv_only

JOB = {}

# modelop.init
def init(job_json: dict) -> None:
    """Store job JSON; input asset paths are provided via job parameters or platform contract."""
    global JOB
    JOB = job_json or {}


# modelop.metrics
def metrics() -> dict:
    """
    Run the ETL pipeline and write CHIP_master.csv, CHIP_baseline.csv, CHIP_comparator.csv.
    Input paths (activity_file, feedback_file, ai_responses_dir) and output_dir are read from
    job_json jobParameters or from implementation assets. Output CSVs are written to output_dir
    (platform may capture this as implementation assets for downstream monitors).
    """
    job = {}
    try:
        raw = JOB.get("rawJson") or "{}"
        job = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        pass
    params = job.get("jobParameters") or {}
    output_dir = params.get("output_dir") or os.environ.get("CHIP_ETL_OUTPUT_DIR", _SCRIPT_DIR)
    activity_file = params.get("activity_file")
    feedback_file = params.get("feedback_file")
    ai_responses_dir = params.get("ai_responses_dir")
    split_method = params.get("split_method", "DATE")
    days_threshold = params.get("days_threshold")
    volume_threshold = int(params.get("volume_threshold", 5000))
    baseline_start_date = params.get("baseline_start_date")
    config_path = params.get("config_path", "config.yaml")
    if os.path.isabs(config_path) or not os.path.exists(config_path):
        config_path = os.path.join(_PARENT, "config.yaml") if os.path.exists(os.path.join(_PARENT, "config.yaml")) else "config.yaml"
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
        config_path=config_path,
        min_records_baseline=min_records_baseline,
        min_records_comparator=min_records_comparator,
    )
    yield result


if __name__ == "__main__":
    # Local run: write CSVs to CHIP_data for downstream monitors
    output_dir = os.path.join(_PARENT, "CHIP_data")
    mock_job = {
        "rawJson": json.dumps({
            "jobParameters": {
                "output_dir": output_dir,
                "config_path": os.path.join(_PARENT, "config.yaml"),
            }
        })
    }
    init(mock_job)
    results = list(metrics())
    print("CHIP_mtr_data monitor output:", results[0] if results else {})
