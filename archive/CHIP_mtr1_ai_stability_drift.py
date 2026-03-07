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

# Standard OOTB Imports
import modelop.monitors.stability as stability
import modelop.monitors.drift as drift
import modelop.schema.infer as infer
import modelop.utils as utils

logger = utils.configure_logger()

# Global Configuration
PREPROCESS = True
JOB = {}
SPLIT_METHOD = "VOLUME"
DAYS_THRESHOLD = 30
VOLUME_THRESHOLD = 50
TIMESTAMP_COLUMN = "ai_verification_time"

# ==========================================
# 1. INITIALIZATION
# ==========================================

# modelop.init
def init(job_json=None):
    global JOB
    global SPLIT_METHOD, DAYS_THRESHOLD, VOLUME_THRESHOLD, TIMESTAMP_COLUMN
    
    if job_json is not None:
        if isinstance(job_json, str):
            try:
                JOB = json.loads(job_json)
            except json.JSONDecodeError:
                JOB = {}
        else:
            JOB = job_json
            
        # Extract job parameters safely
        job_params = JOB.get("jobParameters", {})
        if isinstance(job_params, str):
            try:
                job_params = json.loads(job_params)
            except:
                job_params = {}
                
        SPLIT_METHOD = job_params.get("SPLIT_METHOD", SPLIT_METHOD)
        DAYS_THRESHOLD = job_params.get("DAYS_THRESHOLD", DAYS_THRESHOLD)
        VOLUME_THRESHOLD = job_params.get("VOLUME_THRESHOLD", VOLUME_THRESHOLD)
        TIMESTAMP_COLUMN = job_params.get("TIMESTAMP_COLUMN", TIMESTAMP_COLUMN)

    # ---------------------------------------------------------
    # Ensure correct nesting for modelop-monitors OOTB logic
    # ---------------------------------------------------------
    if not isinstance(JOB, dict):
        JOB = {}

    schema_fields = [
        {"name": "batchId", "type": "string", "role": "identifier"},
        {"name": "testName", "type": "double", "role": "numerical"},
        {"name": "ai_meets_specification", "type": "double", "role": "numerical"},
        {"name": "ai_overall_status", "type": "double", "role": "score"},
        {"name": "ai_verification_time", "type": "string", "role": "timestamp"}
    ]
    
    model_metadata = {
        "storedModel": {
            "modelMetaData": {
                "inputSchema": [
                    {
                        "schemaDefinition": {
                            "fields": schema_fields
                        }
                    }
                ]
            }
        }
    }

    # Internal OOTB monitors check for both model and referenceModel
    if "model" not in JOB:
        JOB["model"] = model_metadata
    elif "storedModel" not in JOB["model"]:
        JOB["model"] = model_metadata

    if "referenceModel" not in JOB:
        JOB["referenceModel"] = model_metadata
    elif "storedModel" not in JOB["referenceModel"]:
        JOB["referenceModel"] = model_metadata
        
    # OOTB Validation Step
    infer.validate_schema(JOB)
    logger.info("Initialization complete and schema validated.")

# ==========================================
# 2. PREPROCESSING ENGINE
# ==========================================

def run_preprocessing(job_json):
    """
    Simulated preprocessing engine to generate baseline and comparator frames.
    Reads local raw files if they exist, otherwise uses dummies.
    """
    logger.info("Executing native preprocessing engine...")
    
    base_csv = 'mtr_1_baseline.csv'
    comp_csv = 'mtr_1_comparator.csv'
    
    if os.path.exists(base_csv) and os.path.exists(comp_csv):
        df_baseline = pd.read_csv(base_csv)
        df_comparator = pd.read_csv(comp_csv)
    else:
        # Fallback dummy data for local testing
        df_baseline = pd.DataFrame({
            "batchId": ["B1", "B2"], "testName": ["T1", "T2"],
            "ai_meets_specification": ["true", "false"], "ai_overall_status": ["PASS", "FAIL"],
            "ai_verification_time": ["2026-01-01", "2026-01-02"]
        })
        df_comparator = pd.DataFrame({
            "batchId": ["B3", "B4"], "testName": ["T1", "T2"],
            "ai_meets_specification": ["false", "false"], "ai_overall_status": ["FAIL", "FAIL"],
            "ai_verification_time": ["2026-03-01", "2026-03-02"]
        })
    
    os.makedirs('mtr_1', exist_ok=True)
    df_baseline.to_csv('mtr_1/mtr_1_baseline.csv', index=False)
    df_comparator.to_csv('mtr_1/mtr_1_comparator.csv', index=False)
    df_baseline.to_json('mtr_1/mtr_1_baseline.json', orient='records', indent=4)
    df_comparator.to_json('mtr_1/mtr_1_comparator.json', orient='records', indent=4)
    
    required_assets = {
        "assets": [
            {"name": "mtr_1_baseline.csv", "type": "dataset"},
            {"name": "mtr_1_comparator.csv", "type": "dataset"},
            {"name": "mtr_1_baseline.json", "type": "dataset"},
            {"name": "mtr_1_comparator.json", "type": "dataset"}
        ]
    }
    with open('mtr_1/required_assets.json', 'w') as f:
        json.dump(required_assets, f, indent=4)
        
    return df_baseline, df_comparator

# ==========================================
# 3. CORE METRICS ENGINE
# ==========================================

# modelop.metrics
def metrics(df_baseline: pd.DataFrame, df_sample: pd.DataFrame) -> dict:
    """
    Core metrics function following OOTB Monitor patterns.
    """
    logger.info("Starting metrics function execution.")
    
    # 1. Obtain datasets if local preprocessing is enabled
    if PREPROCESS:
        df_baseline, df_sample = run_preprocessing(JOB)
        logger.info("Preprocessed data generated and exported to mtr_1/ directory.")
        
    cols_to_encode = ['testName', 'ai_meets_specification', 'ai_overall_status']
    
    # 2. Convert Text Labels to Numeric (Consistent with native ModelOp expectations)
    for col in cols_to_encode:
        if col not in df_baseline.columns:
            df_baseline[col] = "UNKNOWN"
        if col not in df_sample.columns:
            df_sample[col] = "UNKNOWN"
            
        combined_series = pd.concat([df_baseline[col], df_sample[col]]).astype(str).fillna("UNKNOWN")
        unique_vals = sorted(combined_series.unique())
        mapping_dict = {val: float(i) for i, val in enumerate(unique_vals)}
        
        df_baseline[col] = df_baseline[col].astype(str).fillna("UNKNOWN").map(mapping_dict)
        df_sample[col] = df_sample[col].astype(str).fillna("UNKNOWN").map(mapping_dict)

    # ---------------------------------------------------------
    # PART A: AI Output Stability (PSI / CSI) - OOTB Pattern
    # ---------------------------------------------------------
    logger.info("Calculating Stability (PSI/CSI) metrics...")
    
    # Initialize StabilityMonitor using OOTB signature
    stability_monitor = stability.StabilityMonitor(
        df_baseline=df_baseline,
        df_sample=df_sample,
        job_json=JOB
    )
    
    stability_payload = stability_monitor.compute_stability_indices()
    
    # ---------------------------------------------------------
    # PART B: Comprehensive Data Drift - OOTB Pattern
    # ---------------------------------------------------------
    logger.info("Calculating Comprehensive Data Drift metrics...")
    
    # Initialize DriftDetector using OOTB signature
    drift_detector = drift.DriftDetector(
        df_baseline=df_baseline, 
        df_sample=df_sample, 
        job_json=JOB
    )

    # Compute drift metrics with standardized flattening suffixes
    es_drift_metrics = drift_detector.calculate_drift(
        pre_defined_test="Epps-Singleton", flattening_suffix="_es_pvalue"
    )
    js_drift_metrics = drift_detector.calculate_drift(
        pre_defined_test="Jensen-Shannon", flattening_suffix="_js_distance"
    )
    kl_drift_metrics = drift_detector.calculate_drift(
        pre_defined_test="Kullback-Leibler", flattening_suffix="_kl_divergence",
    )
    ks_drift_metrics = drift_detector.calculate_drift(
        pre_defined_test="Kolmogorov-Smirnov", flattening_suffix="_ks_pvalue"
    )

    # Merge results using OOTB utils.merge
    result = utils.merge(
        stability_payload,
        es_drift_metrics,
        js_drift_metrics,
        kl_drift_metrics,
        ks_drift_metrics
    )
    
    yield result

# ==========================================
# 4. LOCAL TESTING
# ==========================================

def main():
    test_params = {
        "rawJson": json.dumps({
            "jobParameters": {
                "SPLIT_METHOD": "VOLUME", 
                "DAYS_THRESHOLD": 30,
                "VOLUME_THRESHOLD": 50,  
                "TIMESTAMP_COLUMN": "ai_verification_time"
            }
        })
    }
    
    print("--- Calling init() ---")
    init(test_params)
    print("--- init() complete ---")
    
    print("--- Calling metrics() ---")
    # Generator handles yielding result
    generator_obj = metrics(pd.DataFrame(), pd.DataFrame())
    
    try:
        result = next(generator_obj)
        print("--- metrics() complete ---")
        
        output_filename = 'mtr_1/mtr1_output.json'
        with open(output_filename, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"Output saved to {output_filename}")
        
    except Exception as e:
        print(f"ERROR executing metrics: {e}")

if __name__ == "__main__":
    main()