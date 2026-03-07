"""
ModelOp Center Monitor 1: AI Output Stability (PSI) & Data Drift
----------------------------------------------------------------
Tracks the behavior and drift of the Claude AI model's outputs.
Merges OOTB Stability Analysis and Comprehensive Data Drift.

Best Practice: Uses infer.validate_schema() to read the 
schema asset (e.g., blank CSV) attached in the ModelOp UI.
"""

import pandas as pd
import json
import modelop.monitors.stability as stability
import modelop.monitors.drift as drift
import modelop.schema.infer as infer
import modelop.utils as utils
import sys

logger = utils.configure_logger()

JOB = {}
GROUP = None

# modelop.init
def init(job_json: dict) -> None:
    """
    Initializes the job, extracts group information, and validates schema fail-fast.
    """
    global JOB
    global GROUP
    
    # Extract job_json and validate schema using the attached UI asset
    JOB = job_json
    infer.validate_schema(job_json)
    
    # Extract GROUP specifically for stability analysis
    try:
        job = json.loads(job_json.get("rawJson", "{}"))
        GROUP = job.get('referenceModel', {}).get('group', None)
    except Exception as e:
        logger.warning(f"Could not extract GROUP from rawJson: {e}")
        GROUP = None

# modelop.metrics
def metrics(df_baseline: pd.DataFrame, df_sample: pd.DataFrame) -> dict:
    """
    Computes combined stability and data drift metrics.
    """
    
    # 1. Initialize & Compute Stability Metrics (PSI, CSI)
    stability_monitor = stability.StabilityMonitor(
        df_baseline=df_baseline, 
        df_sample=df_sample, 
        job_json=JOB,
        group=GROUP
    )
    stability_metrics = stability_monitor.compute_stability_indices()

    # 2. Initialize & Compute Comprehensive Drift Metrics
    drift_detector = drift.DriftDetector(
        df_baseline=df_baseline, 
        df_sample=df_sample, 
        job_json=JOB
    )
    
    es_drift = drift_detector.calculate_drift(pre_defined_test="Epps-Singleton", flattening_suffix="_es_pvalue")
    js_drift = drift_detector.calculate_drift(pre_defined_test="Jensen-Shannon", flattening_suffix="_js_distance")
    kl_drift = drift_detector.calculate_drift(pre_defined_test="Kullback-Leibler", flattening_suffix="_kl_divergence")
    ks_drift = drift_detector.calculate_drift(pre_defined_test="Kolmogorov-Smirnov", flattening_suffix="_ks_pvalue")
    summary_drift = drift_detector.calculate_drift(pre_defined_test="Summary")

    # 3. Concatenate and yield final JSON payload
    result = utils.merge(
        stability_metrics,
        es_drift,
        js_drift,
        kl_drift,
        ks_drift,
        summary_drift
    )
    
    yield result


if __name__ == "__main__":
    # Local Testing Execution Block (Slide 38 ModelOp Developer Training)
    # Assumes mtr_preprocess.py has generated the files in the current directory.
    
    print("Testing Monitor 1 locally...")
    
    # 1. Load the mock job JSON to simulate the platform environment
    try:
        with open('modelop_schema.json', 'r') as f:
            mock_schema = json.load(f)
    except FileNotFoundError:
        print("[!] modelop_schema.json not found. Run mtr_preprocess.py first.")
        sys.exit(1)
        
    mock_job = {
        "rawJson": json.dumps({
            "referenceModel": {
                "storedModel": {
                    "modelMetaData": mock_schema
                }
            }
        })
    }
    
    # 2. Call init()
    init(mock_job)
    
    # 3. Load test data
    try:
        df_b = pd.read_json('mtr_1_baseline.json', orient='records')
        df_c = pd.read_json('mtr_1_comparator.json', orient='records')
    except Exception as e:
         print(f"[!] Error loading test data: {e}")
         sys.exit(1)
         
    # 4. Call metrics()
    results = list(metrics(df_b, df_c))
    
    print("\n[SUCCESS] Yielded Metrics Payload:")
    print(json.dumps(results[0], indent=2))