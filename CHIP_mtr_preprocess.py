"""
BMS CHIP: Data Preprocessing & Monitor Asset Generation Pipeline
----------------------------------------------------------------
Transforms raw Data/DB logs and AI responses into full-dimensional,
monitor-ready datasets for ModelOp Center. Baseline/comparator split is
based on ai_verification_time (date or record-volume threshold). All
batch-related dimensions from AI Responses, batch_activity_log, and
ai_feedback are included; monitors pre-filter columns as needed.

Operations:
    1. Derives Human Ground Truth; flattens Claude AI responses with full dimensions.
    2. Enriches batch-level aggregates from activity and feedback logs.
    3. Merges and evaluates records against config (config.yaml or DEFAULT_CONFIG).
    4. Splits into Baseline and Comparator (DATE, VOLUME, or data-driven).
    5. Writes CHIP_data/CHIP_master.csv and .json (with dataset, split_method).
    6. Exports full column set to CHIP_mtr_1, CHIP_mtr_2, CHIP_mtr_3 (always baseline + comparator).
    7. Generates schema, required_assets.json, README per monitor.
"""

import json
import os
import re
import glob
import argparse
import pandas as pd
from datetime import timedelta, timezone

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# ==========================================
# CONFIGURATION
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
    },
    "output_overwrite": {
        "overwrite_readme": False,
        "overwrite_dmn": False,
        "overwrite_modelop_schema": False,
        "overwrite_required_assets": False,
        "overwrite_blank_schema_asset": False
    }
}

def load_config(config_path="config.yaml"):
    if YAML_AVAILABLE and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return DEFAULT_CONFIG


def get_latest_flat_file(directory, pattern):
    """
    Return the path to the latest file in `directory` matching `pattern` (e.g. 'batch_activity_log_*.json').
    Selection is by modification time (mtime). Returns None if no match.
    """
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return None
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_and_merge_with_upsert(file_paths, top_level_key, id_column='batchId'):
    """
    Load multiple JSON files that each have a top-level list key (e.g. 'batch_activity_log')
    and merge into one DataFrame. Later files override earlier ones for rows with the same id_column (upsert).
    file_paths: list of paths to JSON files.
    top_level_key: key in each JSON whose value is a list of records.
    id_column: column used to deduplicate; last occurrence wins.
    """
    if not file_paths:
        return pd.DataFrame()
    dfs = []
    for path in file_paths:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            recs = data.get(top_level_key, []) if isinstance(data, dict) else []
            if recs:
                dfs.append(pd.DataFrame(recs))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    if id_column in combined.columns:
        combined = combined.drop_duplicates(subset=[id_column], keep='last').reset_index(drop=True)
    return combined

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

def _safe_get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d if d is not None else default

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
    """Flatten AI response JSONs and attach all flattenable dimensions (header, document_type, row/item fields)."""
    flattened_rows = []
    for filepath in glob.glob(os.path.join(directory, "*.json")):
        batch_id_fallback = os.path.basename(filepath).split('_')[0]
        try:
            with open(filepath, 'r') as f: data = json.load(f)
        except json.JSONDecodeError:
            continue

        # ---- BG ----
        if "headerData" in data and "rows" in data:
            h = data.get("headerData", {})
            batch_id = h.get("batch_number", batch_id_fallback)
            ver_time = _safe_get(data, "summary", "verification_time") or ""
            doc_type = "BG"
            for row in data.get("rows", []):
                rdata = row.get("data", {})
                val = rdata.get("overall_batch_result", "Unknown")
                rec = {
                    "businessKey": f"{batch_id}-BG-ROW{row.get('row_id')}",
                    "batchId": batch_id,
                    "ai_verification_time": ver_time,
                    "ai_overall_status": normalize_ai_status(val),
                    "testName": "BG_Material_Check",
                    "ai_meets_specification": str(val),
                    "document_type": doc_type,
                    "material_id": h.get("material_id"),
                    "plant_id": h.get("plant_id"),
                    "batch_number": h.get("batch_number"),
                    "generated_by": h.get("generated_by"),
                    "date_generated": h.get("date_generated"),
                    "system": h.get("system"),
                    "material_id_validation": h.get("material_id_validation"),
                    "plant_id_validation_result": h.get("plant_id_validation_result"),
                    "batch_number_validation_result": h.get("batch_number_validation_result"),
                    "dom_validation_result": h.get("dom_validation_result"),
                    "system_validation_result": h.get("system_validation_result"),
                    "row_id": row.get("row_id"),
                    "row_material": rdata.get("material"),
                    "row_batch": rdata.get("batch"),
                    "row_inspection_lot": rdata.get("inspection_lot"),
                    "row_usage_decision_code": rdata.get("usage_decision_code"),
                    "row_user_status": rdata.get("user_status"),
                    "row_process_order": rdata.get("process_order"),
                    "row_sled_bbd": rdata.get("sled_bbd"),
                    "row_coi": rdata.get("coi"),
                    "row_ai_status": rdata.get("ai_status"),
                    "row_material_validation_result": rdata.get("material_validation_result"),
                    "row_usage_desc_code_validation_status": rdata.get("usage_desc_code_validation_status"),
                    "row_usage_desc_code_lot_validation_status": rdata.get("usage_desc_code_lot_validation_status"),
                    "row_user_status_validation_result": rdata.get("user_status_validation_result"),
                    "row_coi_pm_validation_result": rdata.get("coi_pm_validation_result"),
                    "row_coi_npm_validation_result": rdata.get("coi_npm_validation_result"),
                }
                flattened_rows.append(rec)

        # ---- CCA ----
        elif "qeList" in data:
            batch_id = batch_id_fallback
            ver_time = data.get("verification_time", "")
            doc_type = "CCA"
            for index, qe in enumerate(data.get("qeList", [])):
                val = qe.get("submissionStatus", "Unknown")
                qe_id = qe.get("qeId", index)
                rec = {
                    "businessKey": f"{batch_id}-CCA-{qe_id}",
                    "batchId": batch_id,
                    "ai_verification_time": ver_time,
                    "ai_overall_status": normalize_ai_status(val),
                    "testName": f"CCA_QE_Check_{qe_id}",
                    "ai_meets_specification": str(val),
                    "document_type": doc_type,
                    "cca_qeId": qe.get("qeId"),
                    "cca_submissionStatus": qe.get("submissionStatus"),
                    "cca_fillingCountry": qe.get("fillingCountry"),
                    "cca_submissionType": qe.get("submissionType"),
                    "cca_plannedSubmissionDate": qe.get("plannedSubmissionDate"),
                    "cca_actualSubmissionDate": qe.get("actualSubmissionDate"),
                    "cca_submissionId": qe.get("submissionId"),
                }
                flattened_rows.append(rec)

        # ---- DA ----
        elif "qe" in data and "validationSummary" in data:
            ver_time = data.get("validationSummary", {}).get("verification_time", "")
            doc_type = "DA"
            qe_list = data.get("qe", [])
            if not qe_list:
                val = data.get("validationSummary", {}).get("overallStatus", "PASS")
                flattened_rows.append({
                    "businessKey": f"{batch_id_fallback}-DA-SUMMARY",
                    "batchId": batch_id_fallback,
                    "ai_verification_time": ver_time,
                    "ai_overall_status": normalize_ai_status(val),
                    "testName": "DA_Summary_Check",
                    "ai_meets_specification": str(val),
                    "document_type": doc_type,
                })
            else:
                for index, qe in enumerate(qe_list):
                    val = qe.get("overall_qe_status", "Unknown")
                    qe_no = qe.get("qe_no", index)
                    rec = {
                        "businessKey": f"{batch_id_fallback}-DA-{qe_no}",
                        "batchId": batch_id_fallback,
                        "ai_verification_time": ver_time,
                        "ai_overall_status": normalize_ai_status(val),
                        "testName": f"DA_QE_Check_{qe_no}",
                        "ai_meets_specification": str(qe.get("qe_status", "Unknown")),
                        "document_type": doc_type,
                        "da_qe_no": qe.get("qe_no"),
                        "da_overall_qe_status": qe.get("overall_qe_status"),
                    }
                    flattened_rows.append(rec)

        # ---- EM ----
        elif "emProduct" in data:
            ver_time = data.get("verificationTime", "")
            doc_type = "EM"
            for prod_idx, prod in enumerate(data.get("emProduct", [])):
                lot_no = prod.get("lotNo", f"LOT{prod_idx}")
                for m_idx, media in enumerate(prod.get("emMedia", [])):
                    val = media.get("mediaUsedExpValidStatus", "Unknown")
                    flattened_rows.append({
                        "businessKey": f"{batch_id_fallback}-EM-{lot_no}-MEDIA-{m_idx}",
                        "batchId": batch_id_fallback,
                        "ai_verification_time": ver_time,
                        "ai_overall_status": normalize_ai_status(val),
                        "testName": f"EM_Media_{media.get('mediaName', m_idx)}",
                        "ai_meets_specification": str(val),
                        "document_type": doc_type,
                        "em_lotNo": lot_no,
                        "em_mediaName": media.get("mediaName"),
                    })
                for s_idx, sample in enumerate(prod.get("emSample", [])):
                    val = sample.get("aiStatus", "Unknown")
                    flattened_rows.append({
                        "businessKey": f"{batch_id_fallback}-EM-{lot_no}-SAMPLE-{s_idx}",
                        "batchId": batch_id_fallback,
                        "ai_verification_time": ver_time,
                        "ai_overall_status": normalize_ai_status(val),
                        "testName": f"EM_Sample_{sample.get('sampleType', s_idx)}",
                        "ai_meets_specification": str(val),
                        "document_type": doc_type,
                        "em_lotNo": lot_no,
                        "em_sampleType": sample.get("sampleType"),
                    })

    return pd.DataFrame(flattened_rows)

# Comment/text sample limits for enrich_batch_dimensions (tunable)
ACTIVITY_COMMENT_LAST_N = 5
ACTIVITY_SNIPPET_MAX_CHARS = 300
ACTIVITY_COMMENT_SAMPLE_MAX_CHARS = 1500
FEEDBACK_TEXT_LAST_N = 5
FEEDBACK_SNIPPET_MAX_CHARS = 300
FEEDBACK_TEXT_SAMPLE_MAX_CHARS = 1500


def _parse_commenter_name(message):
    """Extract commenter name from message like 'Rana Atul commented in Comments'. Returns None if no match."""
    if not message or not isinstance(message, str):
        return None
    m = re.match(r'^(.+?)\s+commented in Comments\s*$', message.strip())
    return m.group(1).strip() if m else None


def _parse_reassigned_to(message):
    """Extract assignee name from message like 'Reassigned to Rana Atul'. Returns None if no match."""
    if not message or not isinstance(message, str):
        return None
    m = re.search(r'Reassigned to (.+?)(?:\s*$|\.)', message.strip())
    return m.group(1).strip() if m else None


def _parse_mentioned_names(new_value):
    """Extract unique names from @[Full Name] mentions in comment body. Returns list of strings."""
    if not new_value or not isinstance(new_value, str):
        return []
    names = re.findall(r'@\[([^\]]+)\]', new_value)
    return list(dict.fromkeys(n.strip() for n in names if n and n.strip()))


def _activity_comment_sample(grp):
    """Build activity_comment_sample from last N rows: message or truncated new_value, joined and truncated."""
    grp = grp.sort_values('timestamp', na_position='last').tail(ACTIVITY_COMMENT_LAST_N)
    snippets = []
    for _, r in grp.iterrows():
        msg = (r.get('message') if pd.notna(r.get('message')) else '') or ''
        newv = (r.get('new_value') if pd.notna(r.get('new_value')) else '') or ''
        raw = (msg if len(str(msg)) <= ACTIVITY_SNIPPET_MAX_CHARS else str(msg)[:ACTIVITY_SNIPPET_MAX_CHARS] + '...') if msg else (str(newv)[:ACTIVITY_SNIPPET_MAX_CHARS] + ('...' if len(str(newv)) > ACTIVITY_SNIPPET_MAX_CHARS else ''))
        if raw.strip():
            snippets.append(raw)
    s = ' | '.join(snippets)
    return s[:ACTIVITY_COMMENT_SAMPLE_MAX_CHARS] if len(s) > ACTIVITY_COMMENT_SAMPLE_MAX_CHARS else s


def _feedback_text_sample(grp):
    """Build feedback_text_sample from last N rows: truncated feedback values joined."""
    grp = grp.sort_values('created_at', na_position='last').tail(FEEDBACK_TEXT_LAST_N)
    snippets = []
    for _, r in grp.iterrows():
        fb = (r.get('feedback') if pd.notna(r.get('feedback')) else '') or ''
        s = str(fb)[:FEEDBACK_SNIPPET_MAX_CHARS] + ('...' if len(str(fb)) > FEEDBACK_SNIPPET_MAX_CHARS else '')
        if s.strip():
            snippets.append(s)
    joined = ' | '.join(snippets)
    return joined[:FEEDBACK_TEXT_SAMPLE_MAX_CHARS] if len(joined) > FEEDBACK_TEXT_SAMPLE_MAX_CHARS else joined


def enrich_batch_dimensions(df_merged, activity_file, feedback_file):
    """Add batch-level aggregates from activity log and ai_feedback (one row per AI record; batch fields repeated)."""
    try:
        with open(activity_file, 'r') as f: df_act = pd.DataFrame(json.load(f)['batch_activity_log'])
        with open(feedback_file, 'r') as f: df_fb = pd.DataFrame(json.load(f)['ai_feedback'])
    except FileNotFoundError:
        return df_merged

    if 'batch_number' in df_act.columns: df_act = df_act.rename(columns={'batch_number': 'batchId'})
    if 'batch_id' in df_fb.columns: df_fb = df_fb.rename(columns={'batch_id': 'batchId'})

    # Ensure text columns exist for comment sampling and activity parsing
    for c in ('message', 'new_value', 'old_value'):
        if c not in df_act.columns: df_act[c] = ''
    df_act['message'] = df_act['message'].fillna('').astype(str)
    df_act['new_value'] = df_act['new_value'].fillna('').astype(str)
    df_act['old_value'] = df_act['old_value'].fillna('').astype(str)
    if 'timestamp' not in df_act.columns: df_act['timestamp'] = pd.NaT
    if 'e_signed' not in df_act.columns: df_act['e_signed'] = False
    if 'user_id' not in df_act.columns: df_act['user_id'] = pd.NA
    if 'category' not in df_act.columns: df_act['category'] = ''
    if 'field_name' not in df_act.columns: df_act['field_name'] = ''

    # Activity aggregates per batch
    act_agg = df_act.groupby('batchId').agg(
        activity_event_count=('id', 'count'),
        first_activity_timestamp=('timestamp', 'min'),
        last_activity_timestamp=('timestamp', 'max'),
    ).reset_index()
    act_cats = df_act.groupby('batchId')['category'].apply(lambda x: '|'.join(sorted(x.astype(str).unique()))).reset_index().rename(columns={'category': 'activity_categories_seen'})
    act_fields = df_act.groupby('batchId')['field_name'].apply(lambda x: '|'.join(sorted(x.astype(str).unique()))).reset_index().rename(columns={'field_name': 'activity_field_names_seen'})
    act_agg = act_agg.merge(act_cats, on='batchId', how='left').merge(act_fields, on='batchId', how='left')
    # Activity comment sample and count
    act_comment = df_act.groupby('batchId', group_keys=False).apply(_activity_comment_sample, include_groups=False).reset_index().rename(columns={0: 'activity_comment_sample'})
    act_agg = act_agg.merge(act_comment, on='batchId', how='left')
    act_count = df_act.groupby('batchId', group_keys=False).apply(
        lambda g: ((g['message'].astype(str).str.strip() != '') | (g['new_value'].astype(str).str.strip() != '')).sum(),
        include_groups=False
    ).reset_index().rename(columns={0: 'activity_comment_count'})
    act_agg = act_agg.merge(act_count, on='batchId', how='left')
    act_agg['activity_comment_count'] = act_agg['activity_comment_count'].fillna(0).astype(int)
    ai_verified_counts = df_act[df_act['category'] == 'ai-verified'].groupby('batchId').size()
    failed_counts = df_act[df_act['category'] == 'failed'].groupby('batchId').size()
    act_agg['has_ai_verified'] = act_agg['batchId'].map(lambda b: ai_verified_counts.get(b, 0) > 0)
    act_agg['has_failed'] = act_agg['batchId'].map(lambda b: failed_counts.get(b, 0) > 0)

    # ---- Batch-level e-sign tracking ----
    batch_has_e_signed = (df_act['e_signed'] == True) | (df_act['category'] == 'e-sign-successful')
    e_signed_batches = df_act.loc[batch_has_e_signed].groupby('batchId').size()
    act_agg['batch_e_signed'] = act_agg['batchId'].map(lambda b: e_signed_batches.get(b, 0) > 0)
    esign_events = df_act[df_act['category'] == 'e-sign-successful'].sort_values('timestamp')
    if not esign_events.empty:
        latest_esign = esign_events.groupby('batchId', as_index=False).last()
        act_agg = act_agg.merge(
            latest_esign[['batchId', 'user_id', 'message']].rename(columns={'user_id': 'e_signer_user_id', 'message': '_esign_message'}),
            on='batchId', how='left'
        )
        # e_signer_name: e-sign message rarely contains name; leave null unless we add a lookup later
        act_agg['e_signer_name'] = None
        act_agg = act_agg.drop(columns=['_esign_message'], errors='ignore')
    else:
        act_agg['e_signer_user_id'] = pd.NA
        act_agg['e_signer_name'] = None

    # ---- Batch assignee: current and previous from latest batch_assignee event ----
    assignee_events = df_act[df_act['field_name'] == 'batch_assignee'].sort_values('timestamp')
    if not assignee_events.empty:
        latest_assignee = assignee_events.groupby('batchId', as_index=False).last()
        latest_assignee = latest_assignee.rename(columns={'new_value': 'current_assignee_name', 'old_value': 'previous_assignee_name'})[['batchId', 'current_assignee_name', 'previous_assignee_name']]
        act_agg = act_agg.merge(latest_assignee, on='batchId', how='left')
    else:
        act_agg['current_assignee_name'] = None
        act_agg['previous_assignee_name'] = None
    if 'batch_assignee_id' in df_act['field_name'].values:
        assignee_id_events = df_act[df_act['field_name'] == 'batch_assignee_id'].sort_values('timestamp')
        if not assignee_id_events.empty:
            latest_assignee_id = assignee_id_events.groupby('batchId', as_index=False).last()[['batchId', 'new_value']].rename(columns={'new_value': 'current_assignee_id'})
            act_agg = act_agg.merge(latest_assignee_id, on='batchId', how='left')
    if 'current_assignee_id' not in act_agg.columns:
        act_agg['current_assignee_id'] = None

    # ---- Activity commenter names (from "X commented in Comments") and @mentions ----
    user_comment_df = df_act[df_act['category'] == 'user-comment'].copy()
    if not user_comment_df.empty:
        user_comment_df['_commenter'] = user_comment_df['message'].map(_parse_commenter_name)
        commenters = user_comment_df[user_comment_df['_commenter'].notna()].groupby('batchId')['_commenter'].apply(lambda x: '|'.join(sorted(set(str(n) for n in x)))).reset_index().rename(columns={'_commenter': 'activity_commenter_names'})
        act_agg = act_agg.merge(commenters, on='batchId', how='left')
        all_mentioned = []
        for _, r in user_comment_df.iterrows():
            for name in _parse_mentioned_names(r.get('new_value')):
                all_mentioned.append({'batchId': r['batchId'], '_mentioned': name})
        if all_mentioned:
            mentioned_df = pd.DataFrame(all_mentioned).drop_duplicates().groupby('batchId')['_mentioned'].apply(lambda x: '|'.join(sorted(set(x)))).reset_index().rename(columns={'_mentioned': 'activity_mentioned_names'})
            act_agg = act_agg.merge(mentioned_df, on='batchId', how='left')
    if 'activity_commenter_names' not in act_agg.columns:
        act_agg['activity_commenter_names'] = None
    if 'activity_mentioned_names' not in act_agg.columns:
        act_agg['activity_mentioned_names'] = None

    # Feedback aggregates per batch
    if not df_fb.empty:
        if 'feedback' not in df_fb.columns: df_fb['feedback'] = ''
        df_fb['feedback'] = df_fb['feedback'].fillna('').astype(str)
        fb_agg = df_fb.groupby('batchId').agg(
            feedback_event_count=('id', 'count'),
            first_feedback_at=('created_at', 'min'),
            last_feedback_at=('created_at', 'max'),
        ).reset_index()
        fb_actions = df_fb.groupby('batchId')['action'].apply(lambda x: '|'.join(sorted(x.astype(str).unique()))).reset_index().rename(columns={'action': 'feedback_actions'})
        fb_agg = fb_agg.merge(fb_actions, on='batchId', how='left')
        fb_text = df_fb.groupby('batchId', group_keys=False).apply(_feedback_text_sample, include_groups=False).reset_index().rename(columns={0: 'feedback_text_sample'})
        fb_agg = fb_agg.merge(fb_text, on='batchId', how='left')
        fb_count = df_fb.groupby('batchId')['feedback'].apply(lambda x: (x.astype(str).str.strip() != '').sum()).reset_index()
        fb_count.columns = ['batchId', 'feedback_text_count']
        fb_agg = fb_agg.merge(fb_count, on='batchId', how='left')
        fb_agg['feedback_text_count'] = fb_agg['feedback_text_count'].fillna(0).astype(int)
        df_merged = df_merged.merge(fb_agg, on='batchId', how='left')
    df_merged = df_merged.merge(act_agg, on='batchId', how='left')
    # QA reviewer name: use current assignee as proxy (assignee is often the QA reviewer who applied ground truth)
    df_merged['hitl_reviewer_name'] = df_merged.get('current_assignee_name')
    return df_merged

def map_m2_confusion_term(row, m2_config):
    ai = str(row.get('ai_overall_status', ''))
    qa = str(row.get('hitl_qa_decision', ''))
    for term in ['TP', 'FP', 'TN', 'FN']:
        if ai in parse_config_list(m2_config[term].get('ai_overall_status', [])) and qa in parse_config_list(m2_config[term].get('hitl_qa_decision', [])):
            return term
    return 'EXCLUDE'

def compute_threshold_date(df_final, days_threshold=None, min_records_baseline=20, min_records_comparator=20):
    """Return (threshold_date, used_days) for DATE split. If days_threshold is set, use it; else derive so both sides have at least min records."""
    ser = df_final['ai_verification_time'].dropna()
    if ser.empty:
        return None, None
    max_date = ser.max()
    if days_threshold is not None and days_threshold > 0:
        return max_date - timedelta(days=days_threshold), days_threshold
    sorted_dates = ser.sort_values()
    n = len(sorted_dates)
    for i in range(n):
        t = sorted_dates.iloc[i]
        n_base = (ser <= t).sum()
        n_comp = (ser > t).sum()
        if n_base >= min_records_baseline and n_comp >= min_records_comparator:
            return t, None
    # fallback: median index
    idx = max(0, n // 2 - 1)
    return sorted_dates.iloc[idx], None


def compute_threshold_date_by_fraction(df_final, baseline_fraction=0.4):
    """Return (threshold_date, split_method_str) for percentile-based split: first baseline_fraction of time range = baseline."""
    ser = df_final['ai_verification_time'].dropna()
    if ser.empty:
        return None, f"percentile-{baseline_fraction}"
    min_date = ser.min()
    max_date = ser.max()
    if min_date >= max_date:
        return min_date, f"percentile-{baseline_fraction}"
    threshold_date = min_date + (max_date - min_date) * baseline_fraction
    return threshold_date, f"percentile-{baseline_fraction}"


def build_full_schema(all_columns, score_columns=None, label_columns=None):
    """Build inputSchema with all columns; assign role/dataClass/type for ModelOp."""
    score_columns = score_columns or []
    label_columns = label_columns or []
    properties = {}
    for col in all_columns:
        role = "predictor"
        if col in score_columns:
            role = "score"
        elif col in label_columns:
            role = "label"
        elif col == "weight":
            role = "weight"
        dataClass = "categorical" if col in ("ai_overall_status", "hitl_qa_decision", "testName", "document_type") else "categorical"
        if col == "weight":
            dataClass = "numerical"
        try:
            if col != "weight" and ("timestamp" in col or "time" in col or "date" in col or "at" in col):
                dataClass = "datetime"
        except Exception:
            pass
        schema_type = "float" if col == "weight" else "string"
        properties[col] = {"role": role, "dataClass": dataClass, "type": schema_type}
    return {"inputSchema": {"items": {"properties": properties}}}

# ==========================================
# EXPORT
# ==========================================

MONITOR_DMN_FILES = {
    "CHIP_mtr_1": "CHIP_M1_Stability_Drift.dmn",
    "CHIP_mtr_2": "CHIP_M2_Performance.dmn",
    "CHIP_mtr_3": "CHIP_M3_HITL_Stability_Drift.dmn",
}


def _should_write(path, overwrite_flag=False):
    """Write when file is missing, or overwrite is explicitly enabled."""
    return overwrite_flag or (not os.path.exists(path))


def _write_json_if_allowed(path, payload, overwrite_flag=False):
    if _should_write(path, overwrite_flag=overwrite_flag):
        with open(path, 'w') as f:
            json.dump(payload, f, indent=4)
        return True
    return False


def _write_text_if_allowed(path, content, overwrite_flag=False):
    if _should_write(path, overwrite_flag=overwrite_flag):
        with open(path, 'w') as f:
            f.write(content)
        return True
    return False


def _build_monitor_readme(monitor_name, description):
    """Generate monitor-specific user-facing README content."""
    common_assets = """## Required Assets
- **Baseline Data:** Historical records used to establish expected behavior.
- **Comparator Data:** Recent production records used for current evaluation.
- **Schema Asset:** Parsed by `infer.validate_schema()` to map score/label/predictor roles.
"""
    common_notes = """## Data Notes
- Baseline and comparator exports are full-dimensional batch data.
- Monitor scripts pre-filter columns needed by their metrics functions.
- `CHIP_data/CHIP_master.*` is always refreshed by preprocess runs.
"""
    if monitor_name == "CHIP_mtr_1":
        return f"""# CHIP_MTR_1 Monitor

{description}

## What this monitor tells you
- Detects shifts between baseline and comparator for AI outputs and related dimensions.
- Highlights which features are most unstable (CSI) and how that aligns with drift distance (JS).

{common_assets}
## UI Output Interpretation
- **Generic Table:** High-level summary (largest/smallest CSI, overall PSI, date windows).
- **Generic Bar Graph / Horizontal Bar Graph:** Side-by-side CSI (`data1`) and JS distance (`data2`) by feature.
- **Generic Scatter Plot:** CSI vs JS relationship across top features.
- **Generic Pie/Donut:** Comparator AI outcome mix.

{common_notes}
"""
    if monitor_name == "CHIP_mtr_2":
        return f"""# CHIP_MTR_2 Monitor

{description}

## What this monitor tells you
- Measures concordance between AI and HITL labels using classification metrics.
- Surfaces class balance and reviewer-level activity context for interpretability.

{common_assets}
## UI Output Interpretation
- **Generic Table:** Confusion entries, key scores (accuracy/precision/recall/F1/AUC), reviewer/activity totals, date window.
- **Generic Bar Graph / Horizontal Bar Graph:** Primary concordance metrics (`data1`) for quick comparison.
- **Generic Pie/Donut:** Comparator class balance.

## Known Caveat
- AUC can be `null` when comparator contains only one effective class.

{common_notes}
"""
    return f"""# CHIP_MTR_3 Monitor

{description}

## What this monitor tells you
- Tracks reviewer calibration drift and stability changes over time.
- Compares reviewer behavior to team average and summarizes QA feedback volume.

{common_assets}
## UI Output Interpretation
- **Generic Table:** Largest/smallest CSI, overall PSI, team vs reviewer deltas, QA sample count, date windows.
- **Generic Bar Graph / Horizontal Bar Graph:** CSI (`data1`) and JS distance (`data2`) by feature.
- **Generic Scatter Plot:** Feature-level CSI vs JS relationship.
- **Time Line Graph:** Daily rejection rate (`data1`) and review volume (`data2`).
- **Generic Pie/Donut:** Comparator HITL decision mix.

{common_notes}
"""


def export_monitor_assets(
    df_base, df_comp, monitor_name, schema_def, description, cols_to_keep_legend=None,
    overwrite_controls=None, dmn_templates=None
):
    """Export ALL columns to baseline and comparator CSV/JSON; always write both files (even if empty)."""
    os.makedirs(monitor_name, exist_ok=True)
    all_cols_b = [c for c in df_base.columns]
    all_cols_c = [c for c in df_comp.columns]
    all_cols = list(dict.fromkeys(all_cols_b + all_cols_c))
    df_b = df_base.copy() if not df_base.empty else pd.DataFrame(columns=all_cols_b if all_cols_b else all_cols)
    df_c = df_comp.copy() if not df_comp.empty else pd.DataFrame(columns=all_cols_c if all_cols_c else all_cols)
    for c in all_cols:
        if c not in df_b.columns: df_b[c] = None
        if c not in df_c.columns: df_c[c] = None

    # Always write both baseline and comparator
    df_b.to_csv(os.path.join(monitor_name, f'{monitor_name}_baseline.csv'), index=False)
    df_b.to_json(os.path.join(monitor_name, f'{monitor_name}_baseline.json'), orient='records', date_format='iso')
    df_c.to_csv(os.path.join(monitor_name, f'{monitor_name}_comparator.csv'), index=False)
    df_c.to_json(os.path.join(monitor_name, f'{monitor_name}_comparator.json'), orient='records', date_format='iso')

    overwrite_controls = overwrite_controls or {}
    _write_json_if_allowed(
        os.path.join(monitor_name, 'modelop_schema.json'),
        schema_def,
        overwrite_flag=overwrite_controls.get('overwrite_modelop_schema', False),
    )
    schema_cols = list(schema_def.get("inputSchema", {}).get("items", {}).get("properties", {}).keys())
    blank_schema_path = os.path.join(monitor_name, 'blank_schema_asset.csv')
    if _should_write(blank_schema_path, overwrite_flag=overwrite_controls.get('overwrite_blank_schema_asset', False)):
        pd.DataFrame(columns=schema_cols).to_csv(blank_schema_path, index=False)

    required_assets = [
        {"role": "baseline_data", "description": "Historical Baseline Data required for comparison."},
        {"role": "comparator_data", "description": "Recent Production Comparator Data required for evaluation."},
        {"role": "schema", "description": "Blank schema asset to define input/output roles."}
    ]
    _write_json_if_allowed(
        os.path.join(monitor_name, 'required_assets.json'),
        required_assets,
        overwrite_flag=overwrite_controls.get('overwrite_required_assets', False),
    )

    readme_content = _build_monitor_readme(monitor_name, description)
    _write_text_if_allowed(
        os.path.join(monitor_name, 'README.md'),
        readme_content,
        overwrite_flag=overwrite_controls.get('overwrite_readme', False),
    )

    # DMN overwrite hook (active when template content is provided)
    dmn_templates = dmn_templates or {}
    dmn_content = dmn_templates.get(monitor_name)
    dmn_file = MONITOR_DMN_FILES.get(monitor_name)
    if dmn_file and dmn_content:
        _write_text_if_allowed(
            os.path.join(monitor_name, dmn_file),
            dmn_content,
            overwrite_flag=overwrite_controls.get('overwrite_dmn', False),
        )

# ==========================================
# MAIN
# ==========================================

def execute_pipeline(
    split_method='DATE',
    days_threshold=None,
    volume_threshold=5000,
    baseline_start_date=None,
    activity_file='batch_activity_log_202603042226.json',
    feedback_file='ai_feedback_202603042225.json',
    ai_responses_dir='AI Responses',
    config_path='config.yaml',
    min_records_baseline=20,
    min_records_comparator=20,
    overwrite_readme=False,
    overwrite_dmn=False,
    overwrite_modelop_schema=False,
    overwrite_required_assets=False,
    overwrite_blank_schema_asset=False,
    dmn_templates=None,
):
    print("--- Starting MTR Data Preprocessing Pipeline ---\n")
    config = load_config(config_path)

    # Optional: resolve paths from config (latest-file selection by pattern)
    sources = config.get('sources') or {}
    act_dir = sources.get('activity_directory', '.')
    act_pat = sources.get('activity_pattern')
    if act_pat:
        resolved_activity = get_latest_flat_file(act_dir, act_pat)
        if resolved_activity:
            activity_file = resolved_activity
            print(f"[*] Using latest activity file: {activity_file}")
    fb_dir = sources.get('feedback_directory', '.')
    fb_pat = sources.get('feedback_pattern')
    if fb_pat:
        resolved_feedback = get_latest_flat_file(fb_dir, fb_pat)
        if resolved_feedback:
            feedback_file = resolved_feedback
            print(f"[*] Using latest feedback file: {feedback_file}")
    if sources.get('ai_responses_dir') is not None:
        ai_responses_dir = sources.get('ai_responses_dir')

    # Optional: override split parameters from config
    split_cfg = config.get('split') or {}
    if split_cfg:
        if 'days_threshold' in split_cfg:
            days_threshold = split_cfg['days_threshold']
        if 'min_records_baseline' in split_cfg:
            min_records_baseline = split_cfg['min_records_baseline']
        if 'min_records_comparator' in split_cfg:
            min_records_comparator = split_cfg['min_records_comparator']
    overwrite_cfg = config.get('output_overwrite') or {}
    overwrite_readme = overwrite_cfg.get('overwrite_readme', overwrite_readme)
    overwrite_dmn = overwrite_cfg.get('overwrite_dmn', overwrite_dmn)
    overwrite_modelop_schema = overwrite_cfg.get('overwrite_modelop_schema', overwrite_modelop_schema)
    overwrite_required_assets = overwrite_cfg.get('overwrite_required_assets', overwrite_required_assets)
    overwrite_blank_schema_asset = overwrite_cfg.get('overwrite_blank_schema_asset', overwrite_blank_schema_asset)
    overwrite_controls = {
        'overwrite_readme': bool(overwrite_readme),
        'overwrite_dmn': bool(overwrite_dmn),
        'overwrite_modelop_schema': bool(overwrite_modelop_schema),
        'overwrite_required_assets': bool(overwrite_required_assets),
        'overwrite_blank_schema_asset': bool(overwrite_blank_schema_asset),
    }
    dmn_templates = dmn_templates or {}

    df_ground_truth = derive_ground_truth(activity_file, feedback_file)
    df_ai_flattened = process_real_claude_responses(ai_responses_dir)

    if df_ai_flattened.empty:
        print("[!] No AI records processed.")
        return

    df_final = pd.merge(df_ai_flattened, df_ground_truth, on='batchId', how='left')
    df_final['hitl_qa_decision'] = df_final['hitl_qa_decision'].fillna('Pending')
    df_final = enrich_batch_dimensions(df_final, activity_file, feedback_file)
    df_final['cm_term'] = df_final.apply(map_m2_confusion_term, args=(config['monitor_2_performance'],), axis=1)

    df_final['ai_verification_time'] = pd.to_datetime(df_final['ai_verification_time'], format='mixed', utc=True)
    df_final = df_final.sort_values('ai_verification_time').reset_index(drop=True)

    # Explicit numeric weight column (default 1.0). ModelOp stability/drift monitors require a numeric weight;
    # no string column must be used as weight. Business logic can be applied later (e.g. increase weight when
    # remarks indicate policy misalignment, or when certain flags are set). See docs/DATA_INGESTION.md or
    # monitor metrics() docstrings for design examples.
    df_final['weight'] = 1.0

    if baseline_start_date:
        start_dt = pd.to_datetime(baseline_start_date).replace(tzinfo=timezone.utc)
        df_final = df_final[df_final['ai_verification_time'] >= start_dt]

    split_method_upper = split_method.upper()
    if split_method_upper == 'DATE':
        baseline_fraction = (config.get('split') or {}).get('baseline_fraction')
        if baseline_fraction is not None and days_threshold is None:
            threshold_date, split_method_str = compute_threshold_date_by_fraction(df_final, baseline_fraction=baseline_fraction)
            if threshold_date is None:
                threshold_date, _ = compute_threshold_date(df_final, days_threshold=None, min_records_baseline=min_records_baseline, min_records_comparator=min_records_comparator)
                split_method_str = "date-auto"
        else:
            threshold_date, used_days = compute_threshold_date(df_final, days_threshold=days_threshold, min_records_baseline=min_records_baseline, min_records_comparator=min_records_comparator)
            if threshold_date is None:
                threshold_date = df_final['ai_verification_time'].max() - timedelta(days=days_threshold or 30)
                used_days = days_threshold or 30
            n_baseline = (df_final['ai_verification_time'] <= threshold_date).sum()
            if n_baseline == 0 and days_threshold is not None:
                threshold_date, _ = compute_threshold_date(df_final, days_threshold=None, min_records_baseline=min_records_baseline, min_records_comparator=min_records_comparator)
                used_days = None
                print("[*] Fixed-day split would give empty baseline; using data-driven cutoff (date-auto).")
            split_method_str = f"date-{used_days if used_days is not None else 'auto'}"
        df_base_master = df_final[df_final['ai_verification_time'] <= threshold_date].copy()
        df_comp_master = df_final[df_final['ai_verification_time'] > threshold_date].copy()
        print(f"[*] Splitting by date (cutoff: {threshold_date}, method: {split_method_str}); baseline={len(df_base_master)}, comparator={len(df_comp_master)}")
    elif split_method_upper == 'VOLUME':
        split_method_str = f"volume-{volume_threshold}"
        if len(df_final) <= volume_threshold:
            df_base_master = df_final.copy()
            df_comp_master = pd.DataFrame(columns=df_final.columns)
        else:
            df_base_master = df_final.iloc[:volume_threshold].copy()
            df_comp_master = df_final.iloc[volume_threshold:].copy()
        print(f"[*] Splitting by volume (baseline size={volume_threshold}); baseline={len(df_base_master)}, comparator={len(df_comp_master)}")
    else:
        raise ValueError("split_method must be 'DATE' or 'VOLUME'.")

    # Master dataset with dataset + split_method
    os.makedirs('CHIP_data', exist_ok=True)
    df_base_master = df_base_master.copy()
    df_comp_master = df_comp_master.copy()
    df_base_master['dataset'] = 'baseline'
    df_base_master['split_method'] = split_method_str
    df_comp_master['dataset'] = 'comparator'
    df_comp_master['split_method'] = split_method_str
    df_master = pd.concat([df_base_master, df_comp_master], ignore_index=True)
    df_master.to_csv(os.path.join('CHIP_data', 'CHIP_master.csv'), index=False)
    df_master.to_json(os.path.join('CHIP_data', 'CHIP_master.json'), orient='records', date_format='iso')
    print(f"[*] CHIP_data/CHIP_master.csv and CHIP_master.json written (dataset + split_method columns).")

    # Remove auxiliary columns from baseline/comparator for monitor dirs (they already have dataset/split_method in master only; for monitor dirs we don't add dataset/split_method)
    df_base_master = df_base_master.drop(columns=['dataset', 'split_method'], errors='ignore')
    df_comp_master = df_comp_master.drop(columns=['dataset', 'split_method'], errors='ignore')

    all_columns = list(df_base_master.columns)
    full_schema = build_full_schema(all_columns, score_columns=['ai_overall_status', 'hitl_qa_decision'], label_columns=['hitl_qa_decision'])

    # Monitor 1
    m1_allowed = parse_config_list(config['monitor_1_stability'].get('allowed_ai_overall_status', []))
    df_m1_base = df_base_master[df_base_master['ai_overall_status'].isin(m1_allowed)]
    df_m1_comp = df_comp_master[df_comp_master['ai_overall_status'].isin(m1_allowed)]
    export_monitor_assets(
        df_m1_base, df_m1_comp, 'CHIP_mtr_1', full_schema,
        "Tracks AI output stability and drift between baseline and comparator windows.",
        ['ai_overall_status', 'testName', 'ai_meets_specification'],
        overwrite_controls=overwrite_controls,
        dmn_templates=dmn_templates,
    )
    print("  -> CHIP_mtr_1/ assets created.")

    # Monitor 2
    df_m2_base = df_base_master[df_base_master['cm_term'] != 'EXCLUDE']
    df_m2_comp = df_comp_master[df_comp_master['cm_term'] != 'EXCLUDE']
    export_monitor_assets(
        df_m2_base, df_m2_comp, 'CHIP_mtr_2', full_schema,
        "Evaluates AI-vs-HITL concordance using classification metrics and class balance context.",
        ['ai_overall_status', 'hitl_qa_decision'],
        overwrite_controls=overwrite_controls,
        dmn_templates=dmn_templates,
    )
    print("  -> CHIP_mtr_2/ assets created.")

    # Monitor 3
    m3_allowed = parse_config_list(config['monitor_3_calibration'].get('allowed_hitl_qa_decision', []))
    df_m3_base = df_base_master[df_base_master['hitl_qa_decision'].isin(m3_allowed)]
    df_m3_comp = df_comp_master[df_comp_master['hitl_qa_decision'].isin(m3_allowed)]
    export_monitor_assets(
        df_m3_base, df_m3_comp, 'CHIP_mtr_3', full_schema,
        "Tracks HITL reviewer calibration drift, intervention patterns, and decision stability over time.",
        ['hitl_qa_decision', 'hitl_reviewer_id', 'testName'],
        overwrite_controls=overwrite_controls,
        dmn_templates=dmn_templates,
    )
    print("  -> CHIP_mtr_3/ assets created.")
    print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CHIP preprocess pipeline with optional non-data overwrite controls.")
    parser.add_argument("--split-method", choices=["DATE", "VOLUME"], default="DATE")
    parser.add_argument("--days-threshold", type=int, default=None)
    parser.add_argument("--volume-threshold", type=int, default=5000)
    parser.add_argument("--baseline-start-date", type=str, default=None)
    parser.add_argument("--activity-file", type=str, default='batch_activity_log_202603042226.json')
    parser.add_argument("--feedback-file", type=str, default='ai_feedback_202603042225.json')
    parser.add_argument("--ai-responses-dir", type=str, default='AI Responses')
    parser.add_argument("--config-path", type=str, default='config.yaml')
    parser.add_argument("--min-records-baseline", type=int, default=20)
    parser.add_argument("--min-records-comparator", type=int, default=20)
    parser.add_argument("--overwrite-readme", action="store_true", help="Allow README.md overwrite in monitor folders.")
    parser.add_argument("--overwrite-dmn", action="store_true", help="Allow .dmn overwrite when DMN templates are provided.")
    parser.add_argument("--overwrite-modelop-schema", action="store_true", help="Allow modelop_schema.json overwrite.")
    parser.add_argument("--overwrite-required-assets", action="store_true", help="Allow required_assets.json overwrite.")
    parser.add_argument("--overwrite-blank-schema-asset", action="store_true", help="Allow blank_schema_asset.csv overwrite.")
    args = parser.parse_args()

    execute_pipeline(
        split_method=args.split_method,
        days_threshold=args.days_threshold,
        volume_threshold=args.volume_threshold,
        baseline_start_date=args.baseline_start_date,
        activity_file=args.activity_file,
        feedback_file=args.feedback_file,
        ai_responses_dir=args.ai_responses_dir,
        config_path=args.config_path,
        min_records_baseline=args.min_records_baseline,
        min_records_comparator=args.min_records_comparator,
        overwrite_readme=args.overwrite_readme,
        overwrite_dmn=args.overwrite_dmn,
        overwrite_modelop_schema=args.overwrite_modelop_schema,
        overwrite_required_assets=args.overwrite_required_assets,
        overwrite_blank_schema_asset=args.overwrite_blank_schema_asset,
    )
