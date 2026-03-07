"""
BMS CHIP: Data Preprocessing & Monitor Asset Generation Pipeline
----------------------------------------------------------------
This script transforms the raw Data/DB logs into stripped-down, monitor-specific 
datasets required for ingestion into the ModelOp Center platform.

It performs the following operations:
    1. Derives Human Ground Truth & flattens Claude AI responses.
    2. Merges and evaluates records against mapping configurations.
    3. Splits the final dataset into Baseline (older) and Comparator (recent) sets.
    4. Creates separate subdirectories (mtr_1, mtr_2, mtr_3).
    5. Exports full .csv and .json datasets, stripping out unneeded columns, to preserve traceability.
    6. Generates schema, required_assets.json, and README.md in each subdirectory.
"""

import json
import os
import glob
import pandas as pd
from datetime import timedelta, timezone

# ==========================================
# CONFIGURATION SETTINGS
# ==========================================

DEFAULT_CONFIG = {
    "monitor_1_stability": {
        "allowed_ai_overall_status": ["true", "false", "Submitted", "Planned", "PASS", "FAIL"]
    },
    "monitor_3_calibration": {
        "allowed_hitl_qa_decision": ["Approved", "Rejected", "Reprocess", "Pending"]
    },
    "monitor_2_performance": {
        "TP": {"ai_overall_status": ["false", "Planned", "FAIL"], "hitl_qa_decision": ["Rejected", "Reprocess", "Pending"]},
        "FP": {"ai_overall_status": ["false", "Planned", "FAIL"], "hitl_qa_decision": ["Approved"]},
        "TN": {"ai_overall_status": ["true", "Submitted", "PASS"], "hitl_qa_decision": ["Approved", "Pending"]},
        "FN": {"ai_overall_status": ["true", "Submitted", "PASS"], "hitl_qa_decision": ["Rejected", "Reprocess"]}
    }
}

# The base columns that should be included in ALL datasets for traceability.
# We append monitor-specific fields to this list during export.
BASE_COLUMNS = [
    'batchId',
    'businessKey',
    'ai_verification_time',
    'hitl_review_time'
]

def parse_config_list(val):
    if isinstance(val, list): return [str(v).strip() for v in val]
    if isinstance(val, str): return [v.strip() for v in val.split(',')]
    return []

# ==========================================
# EXTRACTION & NORMALIZATION
# ==========================================

def normalize_ai_status(val):
    val_str = str(val).strip().lower()
    if val_str in ['true', 'submitted', 'pass', 'valid', 'yes']: return 'PASS'
    elif val_str in ['false', 'planned', 'fail', 'invalid', 'no']: return 'FAIL'
    return 'FAIL'

def derive_ground_truth(activity_file, feedback_file):
    try:
        with open(activity_file, 'r') as f: df_act = pd.DataFrame(json.load(f)['batch_activity_log'])
        with open(feedback_file, 'r') as f: df_fb = pd.DataFrame(json.load(f)['ai_feedback'])
    except FileNotFoundError:
        return pd.DataFrame(columns=['batchId', 'hitl_qa_decision', 'hitl_reviewer_id', 'hitl_review_time'])

    if 'batch_number' in df_act.columns: df_act = df_act.rename(columns={'batch_number': 'batchId'})
    if 'batch_id' in df_fb.columns: df_fb = df_fb.rename(columns={'batch_id': 'batchId'})

    reprocess_mask = df_act['category'].isin(['failed', 'revalidate', 'document-check-failed'])
    reprocess_batches = df_act[reprocess_mask]['batchId'].unique()
    
    rejected_batches = df_fb[df_fb['feedback_type'] == 'ai-correction']['batchId'].unique() if not df_fb.empty and 'feedback_type' in df_fb.columns else []
    approved_batches = df_act[(df_act['field_name'] == 'batch_status') & (df_act['new_value'].isin(['COMPLETED', 'READY-FOR-APPROVAL']))]['batchId'].unique()
    
    all_batches = set(df_act['batchId'].dropna().unique()).union(set(df_fb['batchId'].dropna().unique()) if not df_fb.empty else set())
    
    gt_records = []
    for batch in all_batches:
        decision, reviewer_id, review_time = "Pending", None, None
        
        if batch in reprocess_batches:
            decision = "Reprocess"
            latest = df_act[(df_act['batchId'] == batch) & reprocess_mask].sort_values('timestamp').iloc[-1]
            reviewer_id, review_time = latest.get('user_id'), latest.get('timestamp')
        elif batch in rejected_batches:
            decision = "Rejected"
            latest = df_fb[(df_fb['batchId'] == batch) & (df_fb['feedback_type'] == 'ai-correction')].sort_values('created_at').iloc[-1]
            reviewer_id, review_time = latest.get('user_id'), latest.get('created_at')
        elif batch in approved_batches:
            decision = "Approved"
            latest = df_act[(df_act['batchId'] == batch) & ((df_act['field_name'] == 'batch_status') & (df_act['new_value'].isin(['COMPLETED', 'READY-FOR-APPROVAL'])))].sort_values('timestamp').iloc[-1]
            reviewer_id, review_time = latest.get('user_id'), latest.get('timestamp')
            
        gt_records.append({
            "batchId": batch,
            "hitl_qa_decision": decision,
            "hitl_reviewer_id": f"USER-{reviewer_id}" if pd.notna(reviewer_id) else "UNKNOWN",
            "hitl_review_time": review_time
        })
    return pd.DataFrame(gt_records)

def process_real_claude_responses(directory="AI Responses"):
    flattened_rows = []
    for filepath in glob.glob(os.path.join(directory, "*.json")):
        batch_id_fallback = os.path.basename(filepath).split('_')[0]
        try:
            with open(filepath, 'r') as f: data = json.load(f)
        except json.JSONDecodeError: continue
            
        if "headerData" in data and "rows" in data:
            batch_id = data.get("headerData", {}).get("batch_number", batch_id_fallback)
            for row in data.get("rows", []):
                val = row.get("data", {}).get("overall_batch_result", "Unknown")
                flattened_rows.append({"businessKey": f"{batch_id}-BG-ROW{row.get('row_id')}", "batchId": batch_id, "ai_verification_time": data.get("summary", {}).get("verification_time"), "ai_overall_status": normalize_ai_status(val), "testName": "BG_Material_Check", "ai_meets_specification": str(val)})
        elif "qeList" in data:
            for index, qe in enumerate(data.get("qeList", [])):
                val = qe.get("submissionStatus", "Unknown")
                qe_id = qe.get("qeId", index)
                flattened_rows.append({"businessKey": f"{batch_id_fallback}-CCA-{qe_id}", "batchId": batch_id_fallback, "ai_verification_time": data.get("verification_time"), "ai_overall_status": normalize_ai_status(val), "testName": f"CCA_QE_Check_{qe_id}", "ai_meets_specification": str(val)})
        elif "qe" in data and "validationSummary" in data:
            ver_time = data.get("validationSummary", {}).get("verification_time")
            if not data.get("qe", []):
                val = data.get("validationSummary", {}).get("overallStatus", "PASS")
                flattened_rows.append({"businessKey": f"{batch_id_fallback}-DA-SUMMARY", "batchId": batch_id_fallback, "ai_verification_time": ver_time, "ai_overall_status": normalize_ai_status(val), "testName": "DA_Summary_Check", "ai_meets_specification": str(val)})
            else:
                for index, qe in enumerate(data.get("qe", [])):
                    val = qe.get("overall_qe_status", "Unknown")
                    qe_no = qe.get("qe_no", index)
                    flattened_rows.append({"businessKey": f"{batch_id_fallback}-DA-{qe_no}", "batchId": batch_id_fallback, "ai_verification_time": ver_time, "ai_overall_status": normalize_ai_status(val), "testName": f"DA_QE_Check_{qe_no}", "ai_meets_specification": str(qe.get("qe_status", "Unknown"))})
        elif "emProduct" in data:
            ver_time = data.get("verificationTime") 
            for prod_idx, prod in enumerate(data.get("emProduct", [])):
                lot_no = prod.get("lotNo", f"LOT{prod_idx}")
                for m_idx, media in enumerate(prod.get("emMedia", [])):
                    val = media.get("mediaUsedExpValidStatus", "Unknown")
                    media_name = media.get("mediaName", m_idx)
                    flattened_rows.append({"businessKey": f"{batch_id_fallback}-EM-{lot_no}-MEDIA-{m_idx}", "batchId": batch_id_fallback, "ai_verification_time": ver_time, "ai_overall_status": normalize_ai_status(val), "testName": f"EM_Media_{media_name}", "ai_meets_specification": str(val)})
                for s_idx, sample in enumerate(prod.get("emSample", [])):
                    val = sample.get("aiStatus", "Unknown")
                    sample_type = sample.get("sampleType", s_idx)
                    flattened_rows.append({"businessKey": f"{batch_id_fallback}-EM-{lot_no}-SAMPLE-{s_idx}", "batchId": batch_id_fallback, "ai_verification_time": ver_time, "ai_overall_status": normalize_ai_status(val), "testName": f"EM_Sample_{sample_type}", "ai_meets_specification": str(val)})
    return pd.DataFrame(flattened_rows)

def map_m2_confusion_term(row, m2_config):
    ai = str(row.get('ai_overall_status', ''))
    qa = str(row.get('hitl_qa_decision', ''))
    for term in ['TP', 'FP', 'TN', 'FN']:
        if ai in parse_config_list(m2_config[term].get('ai_overall_status', [])) and qa in parse_config_list(m2_config[term].get('hitl_qa_decision', [])):
            return term
    return 'EXCLUDE'

# ==========================================
# EXPORT HELPERS
# ==========================================

def export_monitor_assets(df_base, df_comp, monitor_name, schema_def, description, cols_to_keep):
    """Creates directory, saves filtered CSV/JSON, and creates ModelOp required documentation."""
    os.makedirs(monitor_name, exist_ok=True)
    
    # Combine the universal BASE_COLUMNS with the monitor-specific cols_to_keep
    # Use set() to remove any duplicates, then convert back to list
    final_cols = list(set(BASE_COLUMNS + cols_to_keep))
    
    # Safely filter datasets to ONLY the columns that actually exist in the dataframe
    # This prevents KeyError if a column in BASE_COLUMNS or cols_to_keep is missing from df
    final_cols_b = [col for col in final_cols if col in df_base.columns]
    df_b = df_base[final_cols_b].copy() if not df_base.empty else pd.DataFrame(columns=final_cols_b)
    
    final_cols_c = [col for col in final_cols if col in df_comp.columns]
    df_c = df_comp[final_cols_c].copy() if not df_comp.empty else pd.DataFrame(columns=final_cols_c)
    
    # Export Data
    if not df_b.empty:
        df_b.to_csv(os.path.join(monitor_name, f'{monitor_name}_baseline.csv'), index=False)
        df_b.to_json(os.path.join(monitor_name, f'{monitor_name}_baseline.json'), orient='records', date_format='iso')
    
    if not df_c.empty:
        df_c.to_csv(os.path.join(monitor_name, f'{monitor_name}_comparator.csv'), index=False)
        df_c.to_json(os.path.join(monitor_name, f'{monitor_name}_comparator.json'), orient='records', date_format='iso')
    
    # Create ModelOp compatible Schema JSON & Blank CSV
    with open(os.path.join(monitor_name, 'modelop_schema.json'), 'w') as f:
        json.dump(schema_def, f, indent=4)
        
    # Create the blank_schema_asset using ONLY the schema keys so the UI mapper is clean
    schema_cols = list(schema_def.get("inputSchema", {}).get("items", {}).get("properties", {}).keys())
    pd.DataFrame(columns=schema_cols).to_csv(os.path.join(monitor_name, 'blank_schema_asset.csv'), index=False)

    # 1. Generate required_assets.json (Per ModelOp Best Practices)
    required_assets = [
        {"role": "baseline_data", "description": "Historical Baseline Data required for comparison."},
        {"role": "comparator_data", "description": "Recent Production Comparator Data required for evaluation."},
        {"role": "schema", "description": "Blank schema asset to define input/output roles."}
    ]
    with open(os.path.join(monitor_name, 'required_assets.json'), 'w') as f:
        json.dump(required_assets, f, indent=4)

    # 2. Generate standard README.md (Per ModelOp Best Practices)
    readme_content = """# {monitor_name_upper} Monitor

{description}

## Required Assets
- **Baseline Data:** Historical dataset for establishing the baseline.
- **Comparator Data:** Production dataset to be evaluated.
- **Schema Asset:** Used by `infer.validate_schema()` to identify role assignments.

## Execution
1. The `init` function reads the schema asset to identify predictors, scores, and labels.
2. The `metrics` function computes the test results and yields the JSON payload.
""".format(monitor_name_upper=monitor_name.upper(), description=description)

    with open(os.path.join(monitor_name, 'README.md'), 'w') as f:
        f.write(readme_content)

# ==========================================
# MAIN EXECUTION
# ==========================================

def execute_pipeline(days_threshold=30):
    print("--- Starting MTR Data Preprocessing Pipeline ---\n")
    
    df_ground_truth = derive_ground_truth('batch_activity_log_202603042226.json', 'ai_feedback_202603042225.json')
    df_ai_flattened = process_real_claude_responses("AI Responses")
    
    if df_ai_flattened.empty:
        print("[!] No AI records processed.")
        return
        
    df_final = pd.merge(df_ai_flattened, df_ground_truth, on='batchId', how='left')
    df_final['hitl_qa_decision'] = df_final['hitl_qa_decision'].fillna('Pending')
    df_final['cm_term'] = df_final.apply(map_m2_confusion_term, args=(DEFAULT_CONFIG['monitor_2_performance'],), axis=1)

    df_final['ai_verification_time'] = pd.to_datetime(df_final['ai_verification_time'], format='mixed', utc=True)
    df_final = df_final.sort_values('ai_verification_time').reset_index(drop=True)
    
    # Split Data (DATE Method)
    max_date = df_final['ai_verification_time'].max()
    threshold_date = max_date - timedelta(days=days_threshold)
    df_base_master = df_final[df_final['ai_verification_time'] <= threshold_date].copy()
    df_comp_master = df_final[df_final['ai_verification_time'] > threshold_date].copy()

    print(f"[*] Splitting complete. Master Baseline: {len(df_base_master)} rows, Master Comparator: {len(df_comp_master)} rows.")
    print("\n[*] Generating Monitor-specific Subdirectories and Assets...")

    # ---------------------------------------------------------
    # MONITOR 1: AI Output Stability (PSI) & Drift
    # ---------------------------------------------------------
    m1_schema = {
        "inputSchema": {"items": {"properties": {
            "ai_overall_status": {"role": "score", "dataClass": "categorical", "type": "string"},
            "testName": {"role": "predictor", "dataClass": "categorical", "type": "string"},
            "ai_meets_specification": {"role": "predictor", "dataClass": "categorical", "type": "string"}
        }}}
    }
    m1_desc = "Tracks the behavior and drift of the Claude AI model's output over time using Population Stability Index (PSI) and Data Drift methods."
    m1_cols_to_keep = ['ai_overall_status', 'testName', 'ai_meets_specification']
    
    m1_allowed = parse_config_list(DEFAULT_CONFIG['monitor_1_stability']['allowed_ai_overall_status'])
    df_m1_base = df_base_master[df_base_master['ai_overall_status'].isin(m1_allowed)]
    df_m1_comp = df_comp_master[df_comp_master['ai_overall_status'].isin(m1_allowed)]
    
    export_monitor_assets(df_m1_base, df_m1_comp, 'mtr_1', m1_schema, m1_desc, m1_cols_to_keep)
    print("  -> mtr_1/ assets created successfully.")

    # ---------------------------------------------------------
    # MONITOR 2: Operational Approval Concordance (Performance)
    # ---------------------------------------------------------
    m2_schema = {
        "inputSchema": {"items": {"properties": {
            "ai_overall_status": {"role": "score", "dataClass": "categorical", "type": "string"},
            "hitl_qa_decision": {"role": "label", "dataClass": "categorical", "type": "string"}
        }}}
    }
    m2_desc = "Evaluates the operational performance of the AI by calculating Concordance (Accuracy, Precision, Recall) against Human-In-The-Loop Ground Truth."
    m2_cols_to_keep = ['ai_overall_status', 'hitl_qa_decision']

    df_m2_base = df_base_master[df_base_master['cm_term'] != 'EXCLUDE']
    df_m2_comp = df_comp_master[df_comp_master['cm_term'] != 'EXCLUDE']
    
    export_monitor_assets(df_m2_base, df_m2_comp, 'mtr_2', m2_schema, m2_desc, m2_cols_to_keep)
    print("  -> mtr_2/ assets created successfully.")

    # ---------------------------------------------------------
    # MONITOR 3: QA Calibration (HITL Stability)
    # ---------------------------------------------------------
    m3_schema = {
        "inputSchema": {"items": {"properties": {
            "hitl_qa_decision": {"role": "score", "dataClass": "categorical", "type": "string"},
            "hitl_reviewer_id": {"role": "predictor", "dataClass": "categorical", "type": "string"},
            "testName": {"role": "predictor", "dataClass": "categorical", "type": "string"}
        }}}
    }
    m3_desc = "Tracks Human QA behavior drift (e.g., rubber-stamping detection) and human intervention volume changes."
    m3_cols_to_keep = ['hitl_qa_decision', 'hitl_reviewer_id', 'testName']

    m3_allowed = parse_config_list(DEFAULT_CONFIG['monitor_3_calibration']['allowed_hitl_qa_decision'])
    df_m3_base = df_base_master[df_base_master['hitl_qa_decision'].isin(m3_allowed)]
    df_m3_comp = df_comp_master[df_comp_master['hitl_qa_decision'].isin(m3_allowed)]
    
    export_monitor_assets(df_m3_base, df_m3_comp, 'mtr_3', m3_schema, m3_desc, m3_cols_to_keep)
    print("  -> mtr_3/ assets created successfully.")
    print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    execute_pipeline(days_threshold=30)