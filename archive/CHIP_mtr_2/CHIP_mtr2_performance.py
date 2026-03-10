"""
ModelOp Center Monitor 2: Operational Approval Concordance (Performance)
------------------------------------------------------------------------
Evaluates AI predictions against Human Ground Truth (Classification).

Best Practice: Uses infer.validate_schema() to read the 
schema asset (e.g., blank CSV) attached in the ModelOp UI.
Includes automatic binary mapping for strict OOTB classification compatibility.
"""

import pandas as pd
import json
import sys
import modelop.monitors.performance as performance
import modelop.schema.infer as infer
import modelop.utils as utils

logger = utils.configure_logger()

JOB = {}

# modelop.init
def init(job_json: dict) -> None:
    """
    Initializes the job and validates schema fail-fast using the UI asset.
    """
    global JOB
    JOB = job_json
    infer.validate_schema(job_json)

# modelop.metrics
def metrics(dataframe: pd.DataFrame) -> dict:
    """
    Computes binary classification metrics given the merged pipeline dataset.
    """
    df_eval = dataframe.copy()
    
    # PRE-PROCESSING: Convert strings to strict binary classes for the SDK Evaluator
    # Class 1 (Positive) = FAIL / Flagged / Rejected
    # Class 0 (Negative) = PASS / Good / Approved
    
    # Map AI Score (In-place to respect the schema mappings made in the UI)
    if 'ai_overall_status' in df_eval.columns:
        df_eval['ai_overall_status'] = df_eval['ai_overall_status'].apply(
            lambda x: 1 if str(x).strip().upper() == 'FAIL' else 0
        )
    
    # Map Human Label
    if 'hitl_qa_decision' in df_eval.columns:
        df_eval['hitl_qa_decision'] = df_eval['hitl_qa_decision'].apply(
            lambda x: 1 if str(x).strip().upper() in ['REJECTED', 'REPROCESS', 'PENDING'] else 0
        )

    # Initialize ModelEvaluator with our preprocessed dataframe
    model_evaluator = performance.ModelEvaluator(
        dataframe=df_eval, 
        job_json=JOB
    )

    # Compute and yield classification metrics payload
    yield model_evaluator.evaluate_performance(
        pre_defined_metrics="classification_metrics"
    )
    

if __name__ == "__main__":
    # Local Testing Execution Block (Slide 38 ModelOp Developer Training)
    # Assumes mtr_preprocess.py has generated the files in the current directory.
    
    print("Testing Monitor 2 locally...")
    
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
    
    # 3. Load test data (Performance monitors only need the comparator/sample data)
    try:
        df_c = pd.read_json('mtr_2_comparator.json', orient='records')
    except Exception as e:
         print(f"[!] Error loading test data: {e}")
         sys.exit(1)
         
    # 4. Call metrics()
    results = list(metrics(df_c))
    
    print("\n[SUCCESS] Yielded Metrics Payload:")
    print(json.dumps(results[0], indent=2))