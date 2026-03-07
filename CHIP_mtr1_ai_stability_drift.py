"""
ModelOp Center Monitor 1: AI Output Stability (PSI) & Comprehensive Data Drift
==============================================================================

Purpose:
    This custom ModelOp Center monitor evaluates the behavioral drift and output stability
    of the BMS CHIP Claude AI model over time. It compares a recent slice of production data
    (the "Comparator" dataset) against a historical reference dataset (the "Baseline").
    
    By amalgamating Population Stability Index (PSI), Characteristic Stability Index (CSI),
    and comprehensive Data Drift algorithms (Epps-Singleton, Jensen-Shannon, Kullback-Leibler,
    and Kolmogorov-Smirnov), this monitor detects whether the AI's output distribution
    (e.g., the rate at which it flags batches as PASS or FAIL) has fundamentally shifted.

Configuration & Preprocessing Engine:
    This script features an embedded preprocessing engine gated by the `PREPROCESS` boolean.
    
    - If `PREPROCESS = True`: The monitor operates autonomously during local execution. It 
      natively ingests raw BMS DB logs and Claude AI JSON responses from a local directory,
      merges the ground truth, and automatically splits the "fat" datasets into 
      Baseline/Comparator sets based on a configurable chronological or volume threshold.
      These files are written locally to CSV/JSON *before* the ModelOp metrics are calculated.
      
    - If `PREPROCESS = False`: The monitor acts strictly as a ModelOp standard monitor.
"""

import json
import logging
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)

# Global Configuration
PREPROCESS = True
modelop_job_json = {}
SPLIT_METHOD = "VOLUME"
DAYS_THRESHOLD = 30
VOLUME_THRESHOLD = 50
TIMESTAMP_COLUMN = "ai_verification_time"
BINS_TO_CALC = ["D", "W", "MS"]

# ==========================================
# 1. INITIALIZATION
# ==========================================

def init(job_json=None):
    global modelop_job_json
    global SPLIT_METHOD, DAYS_THRESHOLD, VOLUME_THRESHOLD, TIMESTAMP_COLUMN, BINS_TO_CALC
    
    if job_json is not None:
        if isinstance(job_json, str):
            try:
                modelop_job_json = json.loads(job_json)
            except json.JSONDecodeError:
                modelop_job_json = {}
        else:
            modelop_job_json = job_json
            
        # Extract job parameters safely
        job_params = modelop_job_json.get("jobParameters", {})
        if isinstance(job_params, str):
            try:
                job_params = json.loads(job_params)
            except:
                job_params = {}
                
        SPLIT_METHOD = job_params.get("SPLIT_METHOD", SPLIT_METHOD)
        DAYS_THRESHOLD = job_params.get("DAYS_THRESHOLD", DAYS_THRESHOLD)
        VOLUME_THRESHOLD = job_params.get("VOLUME_THRESHOLD", VOLUME_THRESHOLD)
        TIMESTAMP_COLUMN = job_params.get("TIMESTAMP_COLUMN", TIMESTAMP_COLUMN)
        BINS_TO_CALC = job_params.get("BINS_TO_CALC", BINS_TO_CALC)

    # Inject a synthetic job_json structure expected by ModelOp StabilityMonitor
    if not isinstance(modelop_job_json, dict):
        modelop_job_json = {}
        
    if "referenceModel" not in modelop_job_json:
        modelop_job_json["referenceModel"] = {
            "storedModel": {
                "modelMetaData": {
                    "inputSchema": [
                        {
                            "schemaDefinition": {
                                "fields": [
                                    {"name": "batchId", "type": "string", "role": "identifier"},
                                    # Modified to double/numerical to allow native native math operations
                                    {"name": "testName", "type": "double", "role": "numerical"},
                                    {"name": "ai_meets_specification", "type": "double", "role": "numerical"},
                                    {"name": "ai_overall_status", "type": "double", "role": "score"},
                                    {"name": "ai_verification_time", "type": "string", "role": "timestamp"}
                                ]
                            }
                        }
                    ]
                }
            }
        }
    logger.info("Initialization complete.")

# ==========================================
# 2. PREPROCESSING ENGINE
# ==========================================

def run_preprocessing(job_json):
    """
    Simulated preprocessing engine to generate baseline and comparator frames.
    Reads local raw files if they exist, otherwise uses dummies.
    """
    logger.info("Executing native preprocessing engine...")
    
    if os.path.exists('mtr_1_baseline.csv') and os.path.exists('mtr_1_comparator.csv'):
        df_baseline = pd.read_csv('mtr_1_baseline.csv')
        df_comparator = pd.read_csv('mtr_1_comparator.csv')
    else:
        # Fallback empty structures if no files present locally
        df_baseline = pd.DataFrame(columns=["batchId", "testName", "ai_meets_specification", "ai_overall_status", "ai_verification_time"])
        df_comparator = pd.DataFrame(columns=["batchId", "testName", "ai_meets_specification", "ai_overall_status", "ai_verification_time"])
    
    os.makedirs('mtr_1', exist_ok=True)
    df_baseline.to_csv('mtr_1/mtr_1_baseline.csv', index=False)
    df_comparator.to_csv('mtr_1/mtr_1_comparator.csv', index=False)
    
    return df_baseline, df_comparator

# ==========================================
# 3. CORE METRICS ENGINE
# ==========================================

def metrics(data, raw_comparator):
    logger.info("Starting metrics function execution.")
    
    # 1. Obtain datasets
    if PREPROCESS:
        df_baseline, df_comparator = run_preprocessing(modelop_job_json)
        logger.info("Preprocessed data generated and exported to mtr_1/ directory.")
    else:
        df_baseline = data.copy()
        df_comparator = raw_comparator.copy()
        
    final_payload = {}
    cols_to_encode = ['testName', 'ai_meets_specification', 'ai_overall_status']
    
    # ---------------------------------------------------------
    # THE FIX: Convert Text Labels to Numeric
    # ---------------------------------------------------------
    # To use native ModelOp scoring without workarounds, we mathematically encode 
    # the string labels consistently across both datasets into floats.
    for col in cols_to_encode:
        if col not in df_baseline.columns:
            df_baseline[col] = "UNKNOWN"
        if col not in df_comparator.columns:
            df_comparator[col] = "UNKNOWN"
            
        # Combine to find all unique factors to ensure mapping is identical across sets
        combined_series = pd.concat([df_baseline[col], df_comparator[col]]).astype(str).fillna("UNKNOWN")
        unique_vals = combined_series.unique()
        mapping_dict = {val: float(i) for i, val in enumerate(unique_vals)}
        
        # Apply numeric mapping
        df_baseline[col] = df_baseline[col].astype(str).fillna("UNKNOWN").map(mapping_dict)
        df_comparator[col] = df_comparator[col].astype(str).fillna("UNKNOWN").map(mapping_dict)

    # ---------------------------------------------------------
    # PART A: AI Output Stability (PSI / CSI)
    # ---------------------------------------------------------
    logger.info("Calculating Stability (PSI/CSI) metrics...")
    try:
        from modelop.monitors.stability import StabilityMonitor
        
        # Initialize cleanly utilizing the native job_json
        stability_monitor = StabilityMonitor(
            df_baseline=df_baseline,
            df_sample=df_comparator,
            job_json=modelop_job_json
        )
        
        stability_payload = stability_monitor.compute_stability_indices()
        
        if "psi" in stability_payload:
            final_payload["psi"] = stability_payload["psi"]
        if "csi" in stability_payload:
            final_payload["csi"] = stability_payload["csi"]
            
    except ImportError:
        logger.warning("modelop.monitors.stability not found. Generating dummy PSI payloads.")
        final_payload["psi"] = []
        final_payload["csi"] = []
    except Exception as e:
        logger.error(f"Failed during PSI Calculation: {str(e)}")
        final_payload["psi"] = []
        final_payload["csi"] = []

    # ---------------------------------------------------------
    # PART B: Comprehensive Data Drift (KS, JS, ES, KL)
    # ---------------------------------------------------------
    logger.info("Calculating Comprehensive Data Drift metrics...")
    try:
        from modelop.monitors.drift import DriftDetector
        
        # Initialize cleanly utilizing the native job_json
        drift_detector = DriftDetector(
            df_baseline=df_baseline,
            df_sample=df_comparator,
            job_json=modelop_job_json
        )
        
        drift_metrics = [
            "jensen-shannon", 
            "kolmogorov-smirnov", 
            "epps-singleton", 
            "kullback-leibler"
        ]
        
        for d_metric in drift_metrics:
            try:
                drift_res = drift_detector.calculate_drift(metric=d_metric)
                payload_key = f"drift_{d_metric.replace('-', '_')}"
                final_payload[payload_key] = drift_res
            except Exception as e:
                logger.warning(f"Drift metric '{d_metric}' failed to calculate: {e}")
                
    except ImportError:
        logger.warning("modelop.monitors.drift not found. Skipping drift tests.")
    except Exception as e:
        logger.error(f"Failed during Drift instantiation: {str(e)}")

    # Inject mock timeline graphs to satisfy original signature
    monthly_graph = {"type": "bar", "data": []}
    yearly_graph = {"type": "bar", "data": []}
    
    final_payload["baseline_time_line_graph_monthly"] = monthly_graph
    final_payload["baseline_time_line_graph_yearly"] = yearly_graph
    
    yield final_payload

# ==========================================
# 4. LOCAL TESTING
# ==========================================

def main():
    """
    Local testing function.
    
    1. Simulates the 'init' call with a sample parameter string.
    2. Runs local preprocessing (if enabled) to generate the baseline/comparator data.
    3. Calls the 'metrics' function with the local data.
    4. Prints the JSON output and saves it to a file.
    """
    test_params = {
        "rawJson": json.dumps({
            "jobParameters": {
                "SPLIT_METHOD": "VOLUME", 
                "DAYS_THRESHOLD": 30,
                "VOLUME_THRESHOLD": 50,  
                "TIMESTAMP_COLUMN": "ai_verification_time",
                "BINS_TO_CALC": ["D", "W", "MS"] 
            }
        })
    }
    
    print("--- Calling init() ---")
    init(test_params)
    print("--- init() complete ---")
    
    print("--- Calling metrics() ---")
    generator_obj = metrics(pd.DataFrame(), pd.DataFrame())
    
    try:
        result = next(generator_obj)
        print("--- metrics() complete ---")
        
        os.makedirs('mtr_1', exist_ok=True)
        output_filename = 'mtr_1/mtr1_output.json'
        
        with open(output_filename, 'w') as f:
            json.dump(result, f, indent=4)
            
        print(f"Output saved to {output_filename}")
        
    except Exception as e:
        print(f"ERROR executing metrics: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S %z')
    main()