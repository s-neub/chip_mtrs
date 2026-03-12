"""
BMS CHIP: Data Preprocessing & Mapping Pipeline
-----------------------------------------------
This script fulfills the requirements outlined in the DATA_MAPPING_STRATEGY.md.
It performs the following operations:
    1. Derives the Human Ground Truth (hitl_qa_decision) using process mining logic (Logics A, B, C).
    2. Loads the raw batch IDs (or extracts them from the DB logs).
    3. Loads REAL Claude JSON responses from the 'AI Responses/' directory (BG, CCA, DA, EM schemas).
    4. Flattens the nested Claude JSON responses into test-level tabular rows.
    5. Normalizes diverse AI strings into binary PASS/FAIL.
    6. Merges the AI predictions and Human Ground Truth.
    7. Loads filtering logic from 'job_parameters.json' (or hardcoded defaults).
    8. Splits the final data into Baseline (old data) and Comparator (new data) for ModelOp.
    9. Exports specific, filtered data subsets for Monitor 1, Monitor 2, and Monitor 3.

Prerequisites:
    pip install pandas numpy
"""

import json
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_DATA_DIR = os.path.join(_REPO_ROOT, "CHIP_mtr_data")

# ==========================================
# CONFIGURATION SETTINGS
# ==========================================

DEFAULT_CONFIG = {
    "monitor_1_stability": {
        "allowed_ai_overall_status": ["PASS", "FAIL"]
    },
    "monitor_3_calibration": {
        "allowed_hitl_qa_decision": ["Approved", "Rejected", "Reprocess", "Pending"]
    },
    "monitor_2_performance": {
        "TP": {
            "ai_overall_status": ["FAIL"],
            "hitl_qa_decision": ["Rejected", "Reprocess", "Pending"]
        },
        "FP": {
            "ai_overall_status": ["FAIL"],
            "hitl_qa_decision": ["Approved"]
        },
        "TN": {
            "ai_overall_status": ["PASS"],
            "hitl_qa_decision": ["Approved", "Pending"]
        },
        "FN": {
            "ai_overall_status": ["PASS"],
            "hitl_qa_decision": ["Rejected", "Reprocess"]
        }
    }
}

def load_job_parameter_blocks(job_parameters_path=os.path.join(_DATA_DIR, "job_parameters.json")):
    """Loads monitor parameter blocks from job_parameters.json or returns hardcoded defaults."""
    if os.path.exists(job_parameters_path):
        try:
            with open(job_parameters_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return {
                    "monitor_1_stability": payload.get("monitor_1_stability", DEFAULT_CONFIG["monitor_1_stability"]),
                    "monitor_2_performance": payload.get("monitor_2_performance", DEFAULT_CONFIG["monitor_2_performance"]),
                    "monitor_3_calibration": payload.get("monitor_3_calibration", DEFAULT_CONFIG["monitor_3_calibration"]),
                }
        except Exception:
            pass
    return DEFAULT_CONFIG

def parse_config_list(val):
    """Helper to parse both YAML lists and comma-separated strings into python lists."""
    if isinstance(val, list):
        return [str(v).strip() for v in val]
    if isinstance(val, str):
        return [v.strip() for v in val.split(',')]
    return []

def print_distribution(series, title):
    """Helper function to print terminal distributions with percentages."""
    print(title)
    if series.empty:
        return
    counts = series.value_counts()
    total = len(series)
    for val, count in counts.items():
        pct = (count / total) * 100
        print(f"{val:<15} {count:>6} \t({pct:.1f}%)")

# ==========================================
# AI NORMALIZATION
# ==========================================

def normalize_ai_status(val):
    """
    Normalizes diverse Claude outputs (true/false, Submitted/Planned, PASS/FAIL)
    into a strict binary PASS or FAIL for ModelOp monitor compatibility.
    """
    val_str = str(val).strip().lower()
    
    # 'Submitted' is considered good (TN/PASS) based on prior mapping logic
    if val_str in ['true', 'submitted', 'pass', 'valid', 'yes']:
        return 'PASS'
    # 'Planned' or 'False' indicates an issue was flagged (TP/FAIL)
    elif val_str in ['false', 'planned', 'fail', 'invalid', 'no']:
        return 'FAIL'
    
    return 'FAIL'  # Default to FAIL (Flagged) if unknown to be safe

# ==========================================
# STEP 1: DERIVE HUMAN GROUND TRUTH
# ==========================================

def derive_ground_truth(activity_file: str, feedback_file: str) -> pd.DataFrame:
    """
    Applies the Process Mining logic defined in DATA_MAPPING_STRATEGY.md
    to derive the 'hitl_qa_decision' label from the DB interaction logs.
    """
    try:
        with open(activity_file, 'r') as f:
            df_act = pd.DataFrame(json.load(f)['batch_activity_log'])
        with open(feedback_file, 'r') as f:
            df_fb = pd.DataFrame(json.load(f)['ai_feedback'])
    except FileNotFoundError as e:
        print(f"[!] Error loading logs: {e}")
        return pd.DataFrame(columns=['batchId', 'hitl_qa_decision', 'hitl_reviewer_id', 'hitl_review_time'])

    if 'batch_number' in df_act.columns:
        df_act = df_act.rename(columns={'batch_number': 'batchId'})
    if 'batch_id' in df_fb.columns:
        df_fb = df_fb.rename(columns={'batch_id': 'batchId'})

    reprocess_mask = df_act['category'].isin(['failed', 'revalidate', 'document-check-failed'])
    reprocess_batches = df_act[reprocess_mask]['batchId'].unique()
    
    rejected_batches = []
    if not df_fb.empty and 'feedback_type' in df_fb.columns:
        rejected_mask = df_fb['feedback_type'] == 'ai-correction'
        rejected_batches = df_fb[rejected_mask]['batchId'].unique()
        
    approved_mask = (df_act['field_name'] == 'batch_status') & (df_act['new_value'].isin(['COMPLETED', 'READY-FOR-APPROVAL']))
    potential_approved_batches = df_act[approved_mask]['batchId'].unique()
    
    all_batches = set(df_act['batchId'].dropna().unique()).union(set(df_fb['batchId'].dropna().unique()) if not df_fb.empty else set())
    
    gt_records = []
    for batch in all_batches:
        decision = "Pending"
        reviewer_id = None
        review_time = None
        
        if batch in reprocess_batches:
            decision = "Reprocess"
            latest_act = df_act[(df_act['batchId'] == batch) & reprocess_mask].sort_values('timestamp').iloc[-1]
            reviewer_id = latest_act.get('user_id')
            review_time = latest_act.get('timestamp')
            
        elif batch in rejected_batches:
            decision = "Rejected"
            latest_fb = df_fb[(df_fb['batchId'] == batch) & (df_fb['feedback_type'] == 'ai-correction')].sort_values('created_at').iloc[-1]
            reviewer_id = latest_fb.get('user_id')
            review_time = latest_fb.get('created_at')
            
        elif batch in potential_approved_batches:
            decision = "Approved"
            latest_act = df_act[(df_act['batchId'] == batch) & approved_mask].sort_values('timestamp').iloc[-1]
            reviewer_id = latest_act.get('user_id')
            review_time = latest_act.get('timestamp')
            
        gt_records.append({
            "batchId": batch,
            "hitl_qa_decision": decision,
            "hitl_reviewer_id": f"USER-{reviewer_id}" if pd.notna(reviewer_id) else "UNKNOWN",
            "hitl_review_time": review_time
        })
        
    df_ground_truth = pd.DataFrame(gt_records)
    print(f"[*] Derived Ground Truth labels for {len(df_ground_truth)} unique batches.")
    return df_ground_truth

# ==========================================
# STEP 2 & 3: PARSE AND FLATTEN REAL CLAUDE DATA
# ==========================================

def process_real_claude_responses(directory: str = os.path.join(_DATA_DIR, "AI Responses")) -> pd.DataFrame:
    """
    Scans the provided directory for Claude JSON output files.
    Detects the schema (BG, CCA, DA, EM) and flattens them into tabular format.
    """
    print(f"[*] Extracting and flattening real Claude JSON responses from '{directory}/'...")
    flattened_rows = []
    
    if not os.path.exists(directory):
        print(f"[!] Warning: Directory '{directory}' not found.")
        return pd.DataFrame()
        
    json_files = glob.glob(os.path.join(directory, "*.json"))
    if not json_files:
        print(f"[!] No JSON files found in '{directory}'.")
        return pd.DataFrame()

    for filepath in json_files:
        filename = os.path.basename(filepath)
        # Fallback batch_id extraction from filename (e.g. ADJ9318_CCA.json -> ADJ9318)
        batch_id_fallback = filename.split('_')[0] 
        
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"  [!] Failed to parse JSON in {filename}")
                continue
                
        # SCHEMA 1: BG (Background Data)
        if "headerData" in data and "rows" in data:
            batch_id = data.get("headerData", {}).get("batch_number", batch_id_fallback)
            ver_time = data.get("summary", {}).get("verification_time")
            
            for row in data.get("rows", []):
                row_data = row.get("data", {})
                raw_status = row_data.get("overall_batch_result", "Unknown")
                
                flattened_rows.append({
                    "businessKey": f"{batch_id}-BG-ROW{row.get('row_id')}",
                    "batchId": batch_id,
                    "ai_verification_time": ver_time,
                    "ai_raw_status": str(raw_status),
                    "ai_overall_status": normalize_ai_status(raw_status),
                    "testName": "BG_Material_Check",
                    "ai_meets_specification": str(raw_status)
                })
                
        # SCHEMA 2: CCA (Quality Events)
        elif "qeList" in data:
            batch_id = batch_id_fallback
            ver_time = data.get("verification_time")
            
            for index, qe in enumerate(data.get("qeList", [])):
                qe_id = qe.get("qeId", f"IDX{index}")
                raw_status = qe.get("submissionStatus", "Unknown")
                
                flattened_rows.append({
                    "businessKey": f"{batch_id}-CCA-{qe_id}",
                    "batchId": batch_id,
                    "ai_verification_time": ver_time,
                    "ai_raw_status": str(raw_status),
                    "ai_overall_status": normalize_ai_status(raw_status),
                    "testName": f"CCA_QE_Check_{qe_id}",
                    "ai_meets_specification": str(raw_status)
                })
                
        # SCHEMA 3: DA (Deviation Assessment)
        elif "qe" in data and "validationSummary" in data:
            batch_id = batch_id_fallback
            ver_time = data.get("validationSummary", {}).get("verification_time")
            qe_list = data.get("qe", [])
            
            if not qe_list:
                # If no deviations are listed, log a single row for the empty list
                summary_status = data.get("validationSummary", {}).get("overallStatus", "PASS")
                flattened_rows.append({
                    "businessKey": f"{batch_id}-DA-SUMMARY",
                    "batchId": batch_id,
                    "ai_verification_time": ver_time,
                    "ai_raw_status": str(summary_status),
                    "ai_overall_status": normalize_ai_status(summary_status),
                    "testName": "DA_Summary_Check",
                    "ai_meets_specification": str(summary_status)
                })
            else:
                for index, qe in enumerate(qe_list):
                    qe_id = qe.get("qe_no", f"IDX{index}")
                    raw_status = qe.get("overall_qe_status", "Unknown")
                    
                    flattened_rows.append({
                        "businessKey": f"{batch_id}-DA-{qe_id}",
                        "batchId": batch_id,
                        "ai_verification_time": ver_time,
                        "ai_raw_status": str(raw_status),
                        "ai_overall_status": normalize_ai_status(raw_status),
                        "testName": f"DA_QE_Check_{qe_id}",
                        "ai_meets_specification": qe.get("qe_status", "Unknown")
                    })

        # SCHEMA 4: EM (Environmental Monitoring)
        elif "emProduct" in data:
            batch_id = batch_id_fallback
            # EM uses camelCase for the time key
            ver_time = data.get("verificationTime") 
            
            for prod_idx, prod in enumerate(data.get("emProduct", [])):
                lot_no = prod.get("lotNo", f"LOT{prod_idx}")
                
                # Check media tests
                for m_idx, media in enumerate(prod.get("emMedia", [])):
                    raw_status = media.get("mediaUsedExpValidStatus", "Unknown")
                    flattened_rows.append({
                        "businessKey": f"{batch_id}-EM-{lot_no}-MEDIA-{m_idx}",
                        "batchId": batch_id,
                        "ai_verification_time": ver_time,
                        "ai_raw_status": str(raw_status),
                        "ai_overall_status": normalize_ai_status(raw_status),
                        "testName": f"EM_Media_{media.get('mediaName', m_idx)}",
                        "ai_meets_specification": str(raw_status)
                    })
                
                # Check sample tests
                for s_idx, sample in enumerate(prod.get("emSample", [])):
                    raw_status = sample.get("aiStatus", "Unknown")
                    flattened_rows.append({
                        "businessKey": f"{batch_id}-EM-{lot_no}-SAMPLE-{s_idx}",
                        "batchId": batch_id,
                        "ai_verification_time": ver_time,
                        "ai_raw_status": str(raw_status),
                        "ai_overall_status": normalize_ai_status(raw_status),
                        "testName": f"EM_Sample_{sample.get('sampleType', s_idx)}",
                        "ai_meets_specification": str(raw_status)
                    })
        else:
            print(f"  [?] Unrecognized schema in {filename}, skipping...")
            
    df_ai = pd.DataFrame(flattened_rows)
    print(f"[*] Extracted {len(df_ai)} flattened testing records from {len(json_files)} files.")
    return df_ai

# ==========================================
# STEP 4: MERGE, FILTER, SPLIT & EXPORT
# ==========================================

def map_m2_confusion_term(row, m2_config):
    """Maps a row to a confusion-matrix term based on job-parameter logic."""
    ai = str(row.get('ai_overall_status', ''))
    qa = str(row.get('hitl_qa_decision', ''))
    
    for term in ['TP', 'FP', 'TN', 'FN']:
        if term in m2_config:
            term_dict = m2_config[term]
            allowed_ai = parse_config_list(term_dict.get('ai_overall_status', []))
            allowed_qa = parse_config_list(term_dict.get('hitl_qa_decision', []))
            
            if ai in allowed_ai and qa in allowed_qa:
                return term
                
    return 'EXCLUDE'

def execute_pipeline(split_method='DATE', days_threshold=30, volume_threshold=5000, baseline_start_date=None):
    print("--- Starting BMS CHIP Data Pipeline ---\n")
    
    param_blocks = load_job_parameter_blocks(os.path.join(_DATA_DIR, "job_parameters.json"))
    
    activity_file = os.path.join(_DATA_DIR, 'batch_activity_log_202603042226.json')
    feedback_file = os.path.join(_DATA_DIR, 'ai_feedback_202603042225.json')
    
    print("[*] Deriving Human Ground Truth from DB Logs...")
    df_ground_truth = derive_ground_truth(activity_file, feedback_file)
    
    df_ai_flattened = process_real_claude_responses(os.path.join(_DATA_DIR, "AI Responses"))
    
    if df_ai_flattened.empty:
        print("\n[!] Pipeline halted: No AI records were successfully processed. Ensure CHIP_mtr_data/AI Responses contains valid JSON files.")
        return
        
    full_ai_file = 'extracted_claude_responses_full.csv'
    df_ai_flattened.to_csv(full_ai_file, index=False)
    print(f"[*] Saved raw flattened AI output to '{full_ai_file}'")
    
    print("[*] Merging AI predictions with Human Ground Truth labels...")
    df_final = pd.merge(df_ai_flattened, df_ground_truth, on='batchId', how='left')
    df_final['hitl_qa_decision'] = df_final['hitl_qa_decision'].fillna('Pending')
    
    m1_allowed = parse_config_list(param_blocks['monitor_1_stability'].get('allowed_ai_overall_status', []))
    m3_allowed = parse_config_list(param_blocks['monitor_3_calibration'].get('allowed_hitl_qa_decision', []))
    df_final['cm_term'] = df_final.apply(map_m2_confusion_term, args=(param_blocks['monitor_2_performance'],), axis=1)

    print(f"[*] Formatting data for Split Method: {split_method.upper()}")
    df_final['ai_verification_time'] = pd.to_datetime(df_final['ai_verification_time'], format='mixed', utc=True)
    df_final = df_final.sort_values('ai_verification_time').reset_index(drop=True)
    
    if baseline_start_date:
        start_dt = pd.to_datetime(baseline_start_date).replace(tzinfo=timezone.utc)
        df_final = df_final[df_final['ai_verification_time'] >= start_dt]

    if split_method.upper() == 'DATE':
        max_date = df_final['ai_verification_time'].max()
        threshold_date = max_date - timedelta(days=days_threshold)
        print(f"[*] Splitting dataset by date (Cutoff Date: {threshold_date.strftime('%Y-%m-%d')})")
        df_baseline_master = df_final[df_final['ai_verification_time'] <= threshold_date].copy()
        df_comp_master = df_final[df_final['ai_verification_time'] > threshold_date].copy()
        
    elif split_method.upper() == 'VOLUME':
        print(f"[*] Splitting dataset by volume (Target Baseline Volume: {volume_threshold}, Total Available: {len(df_final)})")
        if len(df_final) <= volume_threshold:
            df_baseline_master = df_final.copy()
            df_comp_master = pd.DataFrame(columns=df_final.columns)
        else:
            df_baseline_master = df_final.iloc[:volume_threshold].copy()
            df_comp_master = df_final.iloc[volume_threshold:].copy()
    else:
        raise ValueError("Invalid SPLIT_METHOD.")

    # Export Target Data
    df_baseline_master.to_csv('bms_chip_master_baseline.csv', index=False)
    df_comp_master.to_csv('bms_chip_master_comparator.csv', index=False)
    
    df_baseline_master[df_baseline_master['cm_term'] != 'EXCLUDE'].to_csv('bms_chip_m2_baseline.csv', index=False)
    df_comp_master[df_comp_master['cm_term'] != 'EXCLUDE'].to_csv('bms_chip_m2_comparator.csv', index=False)

    print(f"\n--- Pipeline Complete ---")
    print(f"[*] Baseline rows saved:   {len(df_baseline_master)}")
    print(f"[*] Comparator rows saved: {len(df_comp_master)}")

    
    # ---------------------------------------------------------
    # PRINT SUMMARY DISTRIBUTIONS
    # ---------------------------------------------------------
    print("\n========================================================")
    print("📊 DATA DISTRIBUTION SUMMARY FOR MODELOP MONITORS")
    print("========================================================\n")
    
    print("--- MONITOR 1: AI Output Stability (PSI) ---")
    print("Tracks the behavior and drift of the Claude model.")
    print("AI Feature Map: 'ai_overall_status'")
    print("-> BASELINE Set (Older Data):")
    base_m1 = df_baseline_master[df_baseline_master['ai_overall_status'].isin(m1_allowed)]
    print_distribution(base_m1['ai_overall_status'], 'ai_overall_status')
    
    print("-> COMPARATOR Set (Recent Data):")
    comp_m1 = df_comp_master[df_comp_master['ai_overall_status'].isin(m1_allowed)]
    print_distribution(comp_m1['ai_overall_status'], 'ai_overall_status')
    print()

    print("--- MONITOR 3: QA Calibration (HITL Stability) ---")
    print("Tracks human review drift (e.g., rubber-stamping) and volume.")
    print("Label Feature Map: 'hitl_qa_decision'")
    print("-> BASELINE Set (Older Data):")
    base_m3 = df_baseline_master[df_baseline_master['hitl_qa_decision'].isin(m3_allowed)]
    print_distribution(base_m3['hitl_qa_decision'], 'hitl_qa_decision')
    
    print("-> COMPARATOR Set (Recent Data):")
    comp_m3 = df_comp_master[df_comp_master['hitl_qa_decision'].isin(m3_allowed)]
    print_distribution(comp_m3['hitl_qa_decision'], 'hitl_qa_decision')
    print()

    print("--- MONITOR 2: Operational Approval Concordance (Performance) ---")
    print("Evaluates recent predictions against Ground Truth.")
    
    # Function to print CM distributions with percentages
    def print_cm_distribution(df_target, title_label):
        print(title_label)
        if df_target.empty:
             print("  [!] Not enough matched data to calculate Concordance Metrics.")
             return
        
        cm_counts = df_target['cm_term'].value_counts().to_dict()
        total_m2_recs = len(df_target)
        
        tp = cm_counts.get('TP', 0)
        fp = cm_counts.get('FP', 0)
        tn = cm_counts.get('TN', 0)
        fn = cm_counts.get('FN', 0)
        
        pct_tp = (tp/total_m2_recs)*100 if total_m2_recs > 0 else 0
        pct_fp = (fp/total_m2_recs)*100 if total_m2_recs > 0 else 0
        pct_tn = (tn/total_m2_recs)*100 if total_m2_recs > 0 else 0
        pct_fn = (fn/total_m2_recs)*100 if total_m2_recs > 0 else 0

        print(f"  True Positives  (TP) : {tp:<6} \t({pct_tp:.1f}%) \n      -> [AI: FAIL/Flagged & QA: Rejected/Reprocess/Pending] (Successful Halt)")
        print(f"  False Positives (FP) : {fp:<6} \t({pct_fp:.1f}%) \n      -> [AI: FAIL/Flagged & QA: Approved] (AI False Alarm)")
        print(f"  True Negatives  (TN) : {tn:<6} \t({pct_tn:.1f}%) \n      -> [AI: PASS/Good & QA: Approved/Pending] (Successful Pass)")
        print(f"  False Negatives (FN) : {fn:<6} \t({pct_fn:.1f}%) \n      -> [AI: PASS/Good & QA: Rejected/Reprocess] (Critical Risk: AI Missed Missing/Bad Data)")

    df_base_m2 = df_baseline_master[df_baseline_master['cm_term'] != 'EXCLUDE']
    print_cm_distribution(df_base_m2, "Confusion Matrix Mapping (BASELINE SET):")
    
    print("\nConfusion Matrix Mapping (COMPARATOR SET ONLY):")
    df_comp_m2 = df_comp_master[df_comp_master['cm_term'] != 'EXCLUDE']
    print_cm_distribution(df_comp_m2, "")

    print("\n========================================================\n")

if __name__ == "__main__":
    # ==========================================
    # CONFIGURATION OPTIONS (WORKSHEET)
    # ==========================================
    SPLIT_METHOD = 'DATE'
    DAYS_THRESHOLD = 30 
    VOLUME_THRESHOLD = 5000 
    BASELINE_START_DATE = None 
    
    execute_pipeline(
        split_method=SPLIT_METHOD,
        days_threshold=DAYS_THRESHOLD,
        volume_threshold=VOLUME_THRESHOLD,
        baseline_start_date=BASELINE_START_DATE
    )