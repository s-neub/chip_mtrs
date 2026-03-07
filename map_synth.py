"""
BMS CHIP: Data Preprocessing & Mapping Pipeline
-----------------------------------------------
This script fulfills the requirements outlined in the DATA_MAPPING_STRATEGY.md.
It performs the following operations:
    1. Derives the Human Ground Truth (hitl_qa_decision) using process mining logic (Logics A, B, C).
    2. Loads the raw batch IDs (or extracts them from the DB logs).
    3. Generates synthetic Claude JSON responses mapped intelligently to Ground Truth.
    4. Flattens the nested Claude JSON responses into test-level tabular rows.
    5. Merges the AI predictions and Human Ground Truth.
    6. Loads filtering logic from 'config.yaml' (or hardcoded defaults).
    7. Splits the final data into Baseline (old data) and Comparator (new data) for ModelOp.
    8. Exports specific, filtered data subsets for Monitor 1, Monitor 2, and Monitor 3.

Prerequisites:
    pip install pandas numpy pyyaml
"""

import json
import os
import pandas as pd
import numpy as np
import random
import yaml
from datetime import datetime, timedelta, timezone

try:
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# ==========================================
# CONFIGURATION SETTINGS
# ==========================================

# Fallback Configuration based on DB Category Distributions
# This config maps ALL categorical states to ensure nothing is excluded in production.
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

def load_config(config_path="config.yaml"):
    """Loads external YAML config or returns the default hardcoded dictionary."""
    if YAML_AVAILABLE and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
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

    # Identify Batches by Logic Tier
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
# STEP 2: INTELLIGENT CLAUDE DATA GENERATION
# ==========================================

def generate_synthetic_claude_logs(df_gt: pd.DataFrame) -> list:
    """
    Simulates the extraction of Claude JSON responses mapped to Ground Truth labels
    to ensure realistic Confusion Matrix distributions based on business reality
    and project Acceptance Criteria (>= 95% Accuracy).
    """
    print(f"[*] Generating synthetic Claude JSON logs for {len(df_gt)} distinct batches...")
    synthetic_ecs_logs = []
    
    standard_tests = ["Appearance", "Mycoplasma", "Sterility", "Endotoxin", "Cell Viability Percentage"]
    base_time = datetime.now(timezone.utc) - timedelta(days=60)
    
    for row in df_gt.itertuples():
        batch_id = row.batchId
        qa_decision = row.hitl_qa_decision
        
        ver_time = base_time + timedelta(days=random.randint(0, 60), hours=random.randint(0, 24))
        
        # Intelligent Mappings ensuring High TP/TN and Low FP/FN
        if qa_decision == 'Approved':
            is_perfect_pass = random.random() < 0.98  # High TN, low FP (AI falsely alarms 2% of time)
        elif qa_decision == 'Rejected':
            is_perfect_pass = random.random() < 0.05  # High TP, low FN (AI successfully catches error 95% of time)
        elif qa_decision == 'Reprocess':
            is_perfect_pass = random.random() < 0.05  # High TP, low FN (AI successfully halts pipeline on bad data 95% of time)
        else: # Pending
            is_perfect_pass = random.random() < 0.95
            
        analyses = []
        for test in standard_tests:
            if is_perfect_pass:
                meets_spec = "Conforms"
                val_result = True
            else:
                # Fails only ~20% of tests to simulate a localized document error
                if random.random() < 0.20:
                    meets_spec = "Does Not Conform"
                    val_result = False
                else:
                    meets_spec = "Conforms"
                    val_result = True
                    
            analyses.append({
                "testName": test,
                "meetsSpecification": meets_spec,
                "validationResult": val_result
            })
            
        # Hard fail fallback: Ensure at least one test fails if it's supposed to be a flawed document
        if not is_perfect_pass and all(t['validationResult'] for t in analyses):
            analyses[0]['meetsSpecification'] = "Does Not Conform"
            analyses[0]['validationResult'] = False
            
        payload = {
            "coaValidationResponse": {
                "joinId": batch_id,
                "manufactureDate": ver_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "analyses": analyses
            }
        }
        synthetic_ecs_logs.append(payload)
        
    return synthetic_ecs_logs

# ==========================================
# STEP 3: FLATTEN CLAUDE JSON OUTPUTS
# ==========================================

def flatten_claude_responses(claude_logs: list) -> pd.DataFrame:
    """Flattens the nested 'analyses' array from the Claude JSON payloads."""
    print("[*] Flattening Claude JSON responses into test-level tabular format...")
    flattened_rows = []
    
    for log in claude_logs:
        response = log.get("coaValidationResponse", {})
        batch_id = response.get("joinId")
        verification_time = response.get("manufactureDate")
        analyses = response.get("analyses", [])
        
        overall_status = "PASS"
        for test in analyses:
            if test.get("meetsSpecification") == "Does Not Conform" or test.get("validationResult") is False:
                overall_status = "FAIL"
                break
                
        for index, test in enumerate(analyses):
            flattened_rows.append({
                "businessKey": f"{batch_id}-TEST{index}", 
                "batchId": batch_id,                      
                "ai_verification_time": verification_time,
                "ai_overall_status": overall_status,      
                "testName": test.get("testName"),
                "ai_meets_specification": test.get("meetsSpecification")
            })
            
    return pd.DataFrame(flattened_rows)


# ==========================================
# STEP 4: MERGE, FILTER, SPLIT & EXPORT
# ==========================================

def map_m2_confusion_term(row, m2_config):
    """Maps a row to a Confusion Matrix Term based on Config logic."""
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
    
    # Load config silently to preserve clean terminal output
    config = load_config("config.yaml")
    
    batch_ids_file = 'bms_chip_batch_ids.csv'
    activity_file = 'batch_activity_log_202603042226.json'
    feedback_file = 'ai_feedback_202603042225.json'
    
    print("[*] Deriving Human Ground Truth from DB Logs...")
    df_ground_truth = derive_ground_truth(activity_file, feedback_file)
    
    try:
        df_batches = pd.read_csv(batch_ids_file)
        col = 'batch_number' if 'batch_number' in df_batches.columns else df_batches.columns[0]
        batch_ids = df_batches[col].dropna().unique().tolist()
        print(f"[*] Loaded {len(batch_ids)} batch IDs from {batch_ids_file}.")
    except FileNotFoundError:
        print(f"[!] Warning: {batch_ids_file} not found.")
        print(f"[*] Extracting batch IDs directly from DB logs to guarantee 1-to-1 match...")
        batch_ids = df_ground_truth['batchId'].dropna().unique().tolist()
    
    claude_logs = generate_synthetic_claude_logs(df_ground_truth)
    df_ai_flattened = flatten_claude_responses(claude_logs)
    
    full_ai_file = 'synthetic_claude_responses_full.csv'
    df_ai_flattened.to_csv(full_ai_file, index=False)
    print(f"[*] Saved raw flattened AI output to '{full_ai_file}'")
    
    print("[*] Merging AI predictions with Human Ground Truth labels...")
    df_final = pd.merge(df_ai_flattened, df_ground_truth, on='batchId', how='left')
    df_final['hitl_qa_decision'] = df_final['hitl_qa_decision'].fillna('Pending')
    
    # Apply Config Mappings
    m1_allowed = parse_config_list(config['monitor_1_stability'].get('allowed_ai_overall_status', []))
    m3_allowed = parse_config_list(config['monitor_3_calibration'].get('allowed_hitl_qa_decision', []))
    df_final['cm_term'] = df_final.apply(map_m2_confusion_term, args=(config['monitor_2_performance'],), axis=1)

    print(f"[*] Formatting data for Split Method: {split_method.upper()}")
    df_final['ai_verification_time'] = pd.to_datetime(df_final['ai_verification_time'])
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

        print(f"  True Positives  (TP) : {tp:<6} \t({pct_tp:.1f}%) \n      -> [AI: FAIL & QA: Rejected/Reprocess/Pending] (Successful Halt)")
        print(f"  False Positives (FP) : {fp:<6} \t({pct_fp:.1f}%) \n      -> [AI: FAIL & QA: Approved] (AI False Alarm)")
        print(f"  True Negatives  (TN) : {tn:<6} \t({pct_tn:.1f}%) \n      -> [AI: PASS & QA: Approved/Pending] (Successful Pass)")
        print(f"  False Negatives (FN) : {fn:<6} \t({pct_fn:.1f}%) \n      -> [AI: PASS & QA: Rejected/Reprocess] (Critical Risk: AI Missed Missing/Bad Data)")

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
    SPLIT_METHOD = 'VOLUME'
    DAYS_THRESHOLD = 30 
    VOLUME_THRESHOLD = 5000 
    BASELINE_START_DATE = None 
    
    execute_pipeline(
        split_method=SPLIT_METHOD,
        days_threshold=DAYS_THRESHOLD,
        volume_threshold=VOLUME_THRESHOLD,
        baseline_start_date=BASELINE_START_DATE
    )