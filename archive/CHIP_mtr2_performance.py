"""
ModelOp Center Monitor 2: Operational Approval Concordance (Performance)
========================================================================
Purpose:
    Evaluates the operational performance of the BMS CHIP Claude AI model.
    It calculates binary classification metrics (Accuracy, Precision, Recall,
    AUC, F1) by treating the AI's prediction (ai_overall_status) as the model
    score and the Human-In-The-Loop QA decision (hitl_qa_decision) as the
    ground truth label.

Configuration:
    - PREPROCESS: If True, natively ingests raw BMS DB logs and Claude AI JSON 
      responses from the local directory, bypassing ModelOp's standard inputs.
    - BINARY MAPPING: The metrics function automatically maps PASS/FAIL strings
      to 0/1 integers to satisfy standard ML evaluation requirements.

Schema Requirements (modelop_schema.json):
    - 'ai_overall_status' -> {"role": "score", "dataClass": "categorical"}
    - 'hitl_qa_decision'  -> {"role": "label", "dataClass": "categorical"}
"""

import os                                                # Import os to handle file pathing and existence checks.
import json                                              # Import json to parse raw DB payloads and structure mock jobs.
import glob                                              # Import glob to detect and match wildcard AI response files.
import pandas as pd                                      # Import pandas for high-performance data manipulation.
from datetime import timedelta, timezone                 # Import datetime utilities to facilitate chronological splitting.

try:                                                     # Attempt to load the proprietary ModelOp platform libraries.
    import modelop.monitors.performance as performance   # Import the performance module to compute classification metrics.
    import modelop.utils as utils                        # Import utility functions for standardized logging.
    logger = utils.configure_logger()                    # Initialize the standardized ModelOp logger.
except ImportError:                                      # Catch exception if running outside the ModelOp environment.
    print("[!] ModelOp libraries not found. Ensure environment is configured.") # Print warning to standard console.

# ==========================================
# 1. PREPROCESSING ENGINE
# ==========================================

PREPROCESS = True                                        # Toggle controlling whether the script runs raw data preprocessing.

def _normalize_ai_status(val: str) -> str:               # Standardizes diverse AI text classifications into a PASS/FAIL binary.
    val_str = str(val).strip().lower()                   # Convert value to string, strip whitespace, and cast to lowercase.
    if val_str in ['true', 'submitted', 'pass', 'valid', 'yes']: return 'PASS' # Check for known positive validation outcomes.
    elif val_str in ['false', 'planned', 'fail', 'invalid', 'no']: return 'FAIL' # Check for known negative/flagged outcomes.
    return 'FAIL'                                        # Fallback to 'FAIL' to ensure unrecognized anomalies flag safely.

def _derive_ground_truth() -> pd.DataFrame:              # Extracts Human-In-The-Loop decisions from DB logs.
    try:                                                 # Try block to safely load DB extracts from local disk.
        with open('batch_activity_log_202603042226.json', 'r') as f: df_act = pd.DataFrame(json.load(f).get('batch_activity_log', [])) # Load activity.
        with open('ai_feedback_202603042225.json', 'r') as f: df_fb = pd.DataFrame(json.load(f).get('ai_feedback', [])) # Load feedback.
    except FileNotFoundError: return pd.DataFrame(columns=['batchId', 'hitl_qa_decision']) # Return empty DataFrame on failure.

    if 'batch_number' in df_act.columns: df_act = df_act.rename(columns={'batch_number': 'batchId'}) # Standardize batch ID column.
    if 'batch_id' in df_fb.columns: df_fb = df_fb.rename(columns={'batch_id': 'batchId'}) # Standardize batch ID column.

    reprocess_batches = df_act[df_act['category'].isin(['failed', 'revalidate', 'document-check-failed'])]['batchId'].unique() # Isolate reprocessed batches.
    rejected_batches = df_fb[df_fb['feedback_type'] == 'ai-correction']['batchId'].unique() if not df_fb.empty else [] # Isolate rejected batches.
    approved_batches = df_act[(df_act['field_name'] == 'batch_status') & (df_act['new_value'].isin(['COMPLETED', 'READY-FOR-APPROVAL']))]['batchId'].unique() # Isolate approved batches.
    all_batches = set(df_act['batchId'].dropna().unique()).union(set(df_fb['batchId'].dropna().unique()) if not df_fb.empty else set()) # Combine all batches.
    
    gt_records = []                                      # Initialize list for finalized ground truth mapping.
    for batch in all_batches:                            # Iterate over every unique batch ID.
        decision = "Pending"                             # Set default decision state to Pending.
        if batch in reprocess_batches: decision = "Reprocess" # Mark as Reprocess.
        elif batch in rejected_batches: decision = "Rejected" # Mark as Rejected.
        elif batch in approved_batches: decision = "Approved" # Mark as Approved.
        gt_records.append({"batchId": batch, "hitl_qa_decision": decision}) # Map finalized human disposition.
    return pd.DataFrame(gt_records)                      # Return assembled Pandas DataFrame.

def _process_real_claude_responses() -> pd.DataFrame:    # Parses nested JSONs from Claude into a flattened DataFrame.
    flattened_rows = []                                  # Initialize list to hold parsed row dictionaries.
    for filepath in glob.glob(os.path.join("AI Responses", "*.json")): # Iterate through all local JSON files.
        batch_id_fallback = os.path.basename(filepath).split('_')[0] # Extract base batch ID from filename.
        try: 
            with open(filepath, 'r') as f: data = json.load(f) # Open and parse the JSON file safely.
        except json.JSONDecodeError: continue            # Skip file if JSON parsing fails.
            
        if "headerData" in data and "rows" in data:      # Schema Detect: BG (Background Data)
            batch_id = data.get("headerData", {}).get("batch_number", batch_id_fallback) # Extract primary batch ID.
            for row in data.get("rows", []):             # Loop through BG tests.
                val = row.get("data", {}).get("overall_batch_result", "Unknown") # Get prediction.
                flattened_rows.append({"batchId": batch_id, "ai_overall_status": _normalize_ai_status(val)}) # Append row.
        elif "qeList" in data:                           # Schema Detect: CCA (Quality Events)
            for index, qe in enumerate(data.get("qeList", [])): # Loop through quality events.
                val = qe.get("submissionStatus", "Unknown") # Get prediction.
                flattened_rows.append({"batchId": batch_id_fallback, "ai_overall_status": _normalize_ai_status(val)}) # Append row.
        elif "qe" in data and "validationSummary" in data: # Schema Detect: DA (Deviation Assessment)
            if not data.get("qe", []):                   # Handle empty deviation array.
                val = data.get("validationSummary", {}).get("overallStatus", "PASS") # Get summary prediction.
                flattened_rows.append({"batchId": batch_id_fallback, "ai_overall_status": _normalize_ai_status(val)}) # Append row.
            else:                                        # Handle populated deviation array.
                for index, qe in enumerate(data.get("qe", [])): # Loop through deviations.
                    val = qe.get("overall_qe_status", "Unknown") # Get prediction.
                    flattened_rows.append({"batchId": batch_id_fallback, "ai_overall_status": _normalize_ai_status(val)}) # Append row.
        elif "emProduct" in data:                        # Schema Detect: EM (Environmental Monitoring)
            for prod_idx, prod in enumerate(data.get("emProduct", [])): # Loop through products.
                for m_idx, media in enumerate(prod.get("emMedia", [])): # Loop through media arrays.
                    val = media.get("mediaUsedExpValidStatus", "Unknown") # Get prediction.
                    flattened_rows.append({"batchId": batch_id_fallback, "ai_overall_status": _normalize_ai_status(val)}) # Append row.
                for s_idx, sample in enumerate(prod.get("emSample", [])): # Loop through sample arrays.
                    val = sample.get("aiStatus", "Unknown") # Get prediction.
                    flattened_rows.append({"batchId": batch_id_fallback, "ai_overall_status": _normalize_ai_status(val)}) # Append row.
    return pd.DataFrame(flattened_rows)                  # Return flattened tabular dataset.

def _run_preprocessing() -> tuple:                       # Executes data aggregation, mapping, and extraction logic.
    df_ground_truth = _derive_ground_truth()             # Extract Ground Truth from local DB logs.
    df_ai_flattened = _process_real_claude_responses()   # Extract and flatten all AI JSONs.
    
    if df_ai_flattened.empty: raise ValueError("No AI records processed.") # Halt if no AI data maps successfully.
        
    df_final = pd.merge(df_ai_flattened, df_ground_truth, on='batchId', how='left') # Join AI Output with Ground Truth on Batch ID.
    df_final['hitl_qa_decision'] = df_final['hitl_qa_decision'].fillna('Pending') # Impute missing human labels.

    # Monitor 2 requires strict filtering to ensure only rows mapping exactly to the Confusion Matrix are evaluated.
    valid_ai = ['PASS', 'FAIL']                          # Define allowed AI predictions.
    valid_qa = ['Approved', 'Rejected', 'Reprocess', 'Pending'] # Define allowed human dispositions.
    
    df_m2 = df_final[df_final['ai_overall_status'].isin(valid_ai) & df_final['hitl_qa_decision'].isin(valid_qa)].copy() # Filter dataset to exact matches.
    return df_m2                                         # Return the fully processed production DataFrame to the metrics function.

# ==========================================
# 2. MODELOP WRAPPERS
# ==========================================

JOB = {}                                                 # Global dictionary holding job asset definition from ModelOp.

# modelop.init
def init(job_json: dict) -> None:                        # Initializes the monitor execution environment.
    global JOB                                           # Expose global JOB variable to store config.
    JOB = job_json                                       # Assign the incoming job configuration.

# modelop.metrics
def metrics(dataframe: pd.DataFrame) -> dict:            # Primary hook triggering Classification metric computations.
    if PREPROCESS:                                       # If internal preprocessing is enabled...
        dataframe = _run_preprocessing()                 # Override input DataFrame with our locally aggregated logic.
    
    df_eval = dataframe.copy()                           # Create a working copy to prevent mutating the original reference.
    
    # In-place Binary Conversion: Standard evaluators require strict 0/1 integers to compute math (AUC, F1, etc.).
    # Class 1 (Positive) = FAIL / Flagged / Rejected
    # Class 0 (Negative) = PASS / Good / Approved
    if 'ai_overall_status' in df_eval.columns:           # Check if the UI-mapped score column exists.
        df_eval['ai_overall_status'] = df_eval['ai_overall_status'].apply( # Apply lambda to convert categorical strings.
            lambda x: 1 if str(x).strip().upper() == 'FAIL' else 0 # 1 if AI flagged an error, 0 if it passed safely.
        )
    
    if 'hitl_qa_decision' in df_eval.columns:            # Check if the UI-mapped label column exists.
        df_eval['hitl_qa_decision'] = df_eval['hitl_qa_decision'].apply( # Apply lambda to convert categorical strings.
            lambda x: 1 if str(x).strip().upper() in ['REJECTED', 'REPROCESS', 'PENDING'] else 0 # 1 if Human halted batch, 0 if approved.
        )

    model_evaluator = performance.ModelEvaluator(        # Instantiate the OOTB Performance Evaluator.
        dataframe=df_eval,                               # Pass the freshly mapped binary evaluation dataset.
        job_json=JOB                                     # Pass the ModelOp job configuration containing the schema layout.
    )

    yield model_evaluator.evaluate_performance(          # Trigger calculation and yield the final JSON metrics payload.
        pre_defined_metrics="classification_metrics"     # Specifically request the classification block (Accuracy, Precision, Recall, etc.).
    )

# ==========================================
# 3. LOCAL TESTING
# ==========================================

if __name__ == "__main__":                               # Execution block for local testing via command line.
    mock_job = {                                         # Generate a mock ModelOp job JSON using exact nested list structures.
        "rawJson": json.dumps({
            "referenceModel": {
                "storedModel": {
                    "modelMetaData": {
                        "inputSchema": [                 # SDK explicitly requires inputSchema to be a List.
                            {
                                "schemaDefinition": {    # SDK explicitly requires the schemaDefinition key.
                                    "items": {
                                        "properties": {  # Target our exact columns used for MTR 2 schema mapping.
                                            "ai_overall_status": {"role": "score", "dataClass": "categorical", "type": "string"},
                                            "hitl_qa_decision": {"role": "label", "dataClass": "categorical", "type": "string"}
                                        }
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        })
    }
    
    init(mock_job)                                       # Fire the init function with simulated UI config.
    results = list(metrics(pd.DataFrame()))              # Call metrics. PREPROCESS=True auto-generates the real dataframe.
    print(json.dumps(results[0], indent=2))              # Pretty-print the yielded classification JSON dictionary.