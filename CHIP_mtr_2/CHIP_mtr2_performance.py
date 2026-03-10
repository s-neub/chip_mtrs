"""
ModelOp Center Monitor 2: Operational Approval Concordance (Performance)
------------------------------------------------------------------------
Evaluates AI predictions against Human Ground Truth (Classification).

Best Practice: Uses infer.validate_schema() to read the 
schema asset (e.g., blank CSV) attached in the ModelOp UI.
Includes automatic binary mapping for strict OOTB classification compatibility.
"""

import os
import pandas as pd
import json
import sys
import modelop.monitors.performance as performance
import modelop.schema.infer as infer
import modelop.utils as utils

logger = utils.configure_logger()

JOB = {}


def _to_native(x):
    """Convert numpy types to native Python for JSON-safe chart/table payloads."""
    if hasattr(x, 'item'):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_to_native(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_native(v) for k, v in x.items()}
    return x


def _get_date_range(df: pd.DataFrame):
    """Get first and last timestamp from dataframe for firstPredictionDate/lastPredictionDate."""
    if df is None or df.empty:
        return None, None
    date_cols = ['ai_verification_time', 'date_generated', 'first_activity_timestamp', 'last_activity_timestamp', 'hitl_review_time']
    for col in date_cols:
        if col not in df.columns:
            continue
        try:
            s = pd.to_datetime(df[col], errors='coerce').dropna()
            if s.empty:
                continue
            return s.min().isoformat(), s.max().isoformat()
        except Exception:
            continue
    return None, None


# Monitor Output Structure keys for Monitor 2
M2_TABLE_KEY = 'ai_hitl_concordance_summary_table'
M2_BAR_KEY = 'ai_hitl_concordance_bar_graph'
M2_HBAR_KEY = 'ai_hitl_concordance_horizontal_bar_graph'
M2_DONUT_KEY = 'hitl_class_balance_donut_chart'
M2_PIE_KEY = 'hitl_class_balance_pie_chart'
M2_ALLOWED_KEYS = (
    M2_TABLE_KEY, M2_BAR_KEY, M2_HBAR_KEY,
    M2_DONUT_KEY, M2_PIE_KEY
)


def _build_m2_visualizations(result: dict, df_eval: pd.DataFrame) -> dict:
    """Build ModelOp chart/table/donut/pie payloads per Monitor Output Structure (data1/data2)."""
    out = {}
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    values = []
    for k in metric_keys:
        v = result.get(k)
        values.append(_to_native(v) if v is not None else 0)
    out[M2_BAR_KEY] = {
        'title': 'AI vs HITL Concordance Metrics',
        'x_axis_label': 'Performance Metric',
        'y_axis_label': 'Score',
        'rotated': False,
        'data': {'metric_score': values},
        'categories': metric_names
    }
    out[M2_HBAR_KEY] = {
        'title': 'AI vs HITL Concordance Metrics (Horizontal)',
        'x_axis_label': 'Score',
        'y_axis_label': 'Performance Metric',
        'rotated': True,
        'data': {'metric_score': values},
        'categories': metric_names
    }
    rows = []
    cm = result.get('confusion_matrix')
    if cm is None and 'performance' in result and result['performance']:
        cm = result['performance'][0].get('values', {}).get('confusion_matrix')
    if isinstance(cm, list) and cm:
        for i, row_dict in enumerate(cm):
            for actual_k, val in row_dict.items():
                rows.append({'Metric': f'Confusion Pred={i} Actual={actual_k}', 'Value': _to_native(val)})
    for k in metric_keys:
        v = result.get(k)
        if v is not None:
            rows.append({'Metric': k.replace('_', ' ').title(), 'Value': _to_native(v)})
    if 'feedback_event_count' in df_eval.columns:
        total_fb = int(df_eval['feedback_event_count'].fillna(0).sum())
        rows.append({'Metric': 'Total feedback events', 'Value': total_fb})
    if 'activity_comment_count' in df_eval.columns:
        total_act = int(df_eval['activity_comment_count'].fillna(0).sum())
        rows.append({'Metric': 'Total activity comments', 'Value': total_act})
    if 'hitl_reviewer_id' in df_eval.columns:
        vc = df_eval['hitl_reviewer_id'].fillna('Unknown').astype(str).value_counts()
        for rev, cnt in vc.items():
            rows.append({'Metric': 'Reviewer volume', 'Value': f'{str(rev)}: {int(cnt)}'})
    out[M2_TABLE_KEY] = rows
    label_col = 'hitl_qa_decision' if 'hitl_qa_decision' in df_eval.columns else 'ai_overall_status'
    if label_col in df_eval.columns:
        vc = df_eval[label_col].astype(str).value_counts()
        counts = _to_native(vc.tolist())
        cats = _to_native(vc.index.tolist())
        out[M2_DONUT_KEY] = {
            'title': f'Class Balance in Comparator Window ({label_col})',
            'type': 'donut',
            'data': {'decision_count': counts},
            'categories': cats
        }
        out[M2_PIE_KEY] = {
            'title': f'Class Balance in Comparator Window ({label_col})',
            'type': 'pie',
            'data': {'decision_count': counts},
            'categories': cats
        }
    return out


# modelop.init
def init(job_json: dict) -> None:
    """
    Initializes the job and validates schema fail-fast using the UI asset.
    """
    logger = utils.configure_logger()
    # global JOB
    # JOB = job_json
    # infer.validate_schema(job_json)

# modelop.metrics
def metrics(dataframe: pd.DataFrame) -> dict: #type: ignore
    """
    Computes binary classification metrics. Yields only monitor-specific chart/table
    keys. Scalars and dates are included as summary table rows.

    Weight variable: This monitor does not use a weight column for scoring. The pipeline
    still provides a numeric "weight" column (default 1.0) for consistency with stability
    monitors (M1, M3). If you later add weighted performance (e.g. weight by record
    importance), use the same pattern: set weight in preprocess from business logic
    (e.g. increase when remarks indicate policy misalignment or a flag requires_escalation),
    then pass that column into the evaluator if/when the SDK supports it.
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

    result = model_evaluator.evaluate_performance(pre_defined_metrics="classification_metrics")
    viz = _build_m2_visualizations(result, df_eval)
    first_d, last_d = _get_date_range(df_eval)
    if viz.get(M2_TABLE_KEY) is not None:
        if first_d is not None:
            viz[M2_TABLE_KEY].append({'Metric': 'First prediction date', 'Value': first_d})
        if last_d is not None:
            viz[M2_TABLE_KEY].append({'Metric': 'Last prediction date', 'Value': last_d})
    output = {k: viz[k] for k in M2_ALLOWED_KEYS if k in viz}
    yield output
    

if __name__ == "__main__":
    # Local Testing Execution Block (Slide 38 ModelOp Developer Training)
    # Assumes mtr_preprocess.py has generated the files in the current directory.
    
    print("Testing Monitor 2 locally...")
    
    # 1. Load the mock job JSON to simulate the platform environment
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(script_dir, 'modelop_schema.json'), 'r') as f:
            schema_from_file = json.load(f)
    except FileNotFoundError:
        print("[!] modelop_schema.json not found. Run mtr_preprocess.py first.")
        sys.exit(1)

    # Convert preprocess schema (inputSchema.items.properties) to ModelOp infer format (fields array)
    # M2 needs exactly one score (AI prediction) and one label (ground truth): ai_overall_status=score, hitl_qa_decision=label
    props = schema_from_file.get("inputSchema", {}).get("items", {}).get("properties", {})
    fields = []
    for name, p in props.items():
        if name == "ai_overall_status":
            role = "score"
        elif name == "hitl_qa_decision":
            role = "label"
        else:
            role = p.get("role", "predictor")
            if role == "score":
                role = "predictor"
            if role == "non-predictor":
                role = "non_predictor"
        data_class = p.get("dataClass", "categorical")
        if data_class not in ("numerical", "categorical"):
            data_class = "categorical"
        fields.append({
            "name": name,
            "role": role,
            "dataClass": data_class,
            "driftCandidate": role in ("predictor", "non_predictor"),
            "specialValues": [],
            "protectedClass": False,
            "scoringOptional": False,
            "type": "string" if p.get("type") not in ("int", "float", "double", "long", "string", "boolean", "null") else p.get("type", "string")
        })
    mock_schema_def = {"fields": fields}

    # ModelOp expects modelMetaData.inputSchema = [ { "schemaDefinition": <schema> } ]
    mock_job = {
        "rawJson": json.dumps({
            "referenceModel": {
                "storedModel": {
                    "modelMetaData": {
                        "inputSchema": [{"schemaDefinition": mock_schema_def}]
                    }
                }
            }
        })
    }
    
    # 2. Call init()
    init(mock_job)
    
    # 3. Load test data (full columns so schema columns exist; platform may pre-filter when invoking metrics())
    try:
        df_c = pd.read_json(os.path.join(script_dir, 'CHIP_mtr_2_comparator.json'), orient='records')
        if df_c.empty:
            print("[!] Comparator is empty. Run preprocess with data that yields comparator records.")
            sys.exit(1)
    except Exception as e:
         print(f"[!] Error loading test data: {e}")
         sys.exit(1)
         
    # 4. Call metrics() and 5. Always write metrics payload to JSON (even on error, so a file is produced every run)
    out_path = os.path.join(script_dir, 'CHIP_mtr_2_test_results.json')

    def _json_serial(obj):
        if hasattr(obj, 'item'):
            v = obj.item()
            if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')):
                return None
            return v
        if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
            return None
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _nan_to_none(obj):
        import math
        if hasattr(obj, 'item'):
            v = obj.item()
            return None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, dict):
            return {k: _nan_to_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_nan_to_none(x) for x in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    try:
        results = list(metrics(df_c))
        payload = _nan_to_none(results[0])
    except Exception as e:
        payload = {"error": str(e), "metrics_computed": False}
        print(f"[!] metrics() failed: {e}")

    wrapped_payload = [payload]
    with open(out_path, 'w') as f:
        json.dump(wrapped_payload, f, indent=2, default=_json_serial)
    print(f"\n[SUCCESS] Output written to {out_path}")
    print(json.dumps(wrapped_payload, indent=2, default=_json_serial))