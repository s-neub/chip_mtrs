"""
ModelOp Center Monitor 3: QA Calibration (HITL Stability)
---------------------------------------------------------
Tracks human review drift and behavior over time.
Merges OOTB Stability Analysis and Comprehensive Data Drift.

Best Practice: Uses infer.validate_schema() to read the 
schema asset (e.g., blank CSV) attached in the ModelOp UI.
"""

import os
import pandas as pd
import json
import sys
import modelop.monitors.stability as stability
import modelop.monitors.drift as drift
import modelop.schema.infer as infer
import modelop.utils as utils

logger = utils.configure_logger()

JOB = {}
GROUP = None


def _to_native(x):
    """Convert numpy types to native Python for JSON-safe chart/table payloads."""
    if hasattr(x, 'item'):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_to_native(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_native(v) for k, v in x.items()}
    return x


def _pretty_feature_name(name):
    """Convert snake_case feature names into UI-friendly labels."""
    return str(name).replace('_', ' ').strip().title()


def _get_date_range(df: pd.DataFrame):
    """Get first and last timestamp from dataframe for monitor output tracking. Prefers ai_verification_time."""
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


def _compute_reviewer_stats(df: pd.DataFrame):
    """
    Compare single QA reviewer metrics to team. Group by hitl_reviewer_id; compute volume and rejection_rate;
    team average is overall mean rejection rate; return list of {Reviewer, Volume, Rejection Rate, vs Team}.
    """
    if df is None or df.empty:
        return []
    if 'hitl_reviewer_id' not in df.columns or 'hitl_qa_decision' not in df.columns:
        return []
    df = df.copy()
    df['hitl_reviewer_id'] = df['hitl_reviewer_id'].fillna('Unknown').astype(str)
    if not pd.api.types.is_numeric_dtype(df['hitl_qa_decision']):
        df['hitl_qa_decision'] = df['hitl_qa_decision'].apply(
            lambda x: 1.0 if str(x).strip().upper() in ("REJECTED", "REPROCESS", "PENDING") else 0.0
        )
    team_avg = float(df['hitl_qa_decision'].mean())
    grp = df.groupby('hitl_reviewer_id', as_index=False).agg(
        Volume=('hitl_qa_decision', 'count'),
        Rejection_Rate=('hitl_qa_decision', 'mean')
    )
    rows = []
    for _, r in grp.iterrows():
        vs_team = float(r['Rejection_Rate']) - team_avg
        rows.append({
            'Reviewer': _to_native(r['hitl_reviewer_id']),
            'Volume': int(r['Volume']),
            'Rejection Rate': _to_native(round(r['Rejection_Rate'], 4)),
            'vs Team': _to_native(round(vs_team, 4))
        })
    return rows


def _compute_time_series(df: pd.DataFrame):
    """
    Daily rejection rate and volume by date (from hitl_review_time or ai_verification_time).
    Returns dict with time_line_graph payload: data1 = [[date_iso, rejection_rate], ...], data2 = [[date_iso, volume], ...].
    """
    if df is None or df.empty:
        return None
    date_col = None
    for col in ['hitl_review_time', 'ai_verification_time', 'last_activity_timestamp', 'first_activity_timestamp']:
        if col in df.columns:
            date_col = col
            break
    if date_col is None or 'hitl_qa_decision' not in df.columns:
        return None
    df = df.copy()
    df['_dt'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['_dt'])
    if df.empty:
        return None
    df['_date'] = df['_dt'].dt.date
    if not pd.api.types.is_numeric_dtype(df['hitl_qa_decision']):
        df['hitl_qa_decision'] = df['hitl_qa_decision'].apply(
            lambda x: 1.0 if str(x).strip().upper() in ("REJECTED", "REPROCESS", "PENDING") else 0.0
        )
    daily = df.groupby('_date', as_index=False).agg(
        rejection_rate=('hitl_qa_decision', 'mean'),
        volume=('hitl_qa_decision', 'count')
    ).sort_values('_date')
    daily['_date_iso'] = daily['_date'].astype(str)
    data1 = [[row['_date_iso'], _to_native(round(row['rejection_rate'], 4))] for _, row in daily.iterrows()]
    data2 = [[row['_date_iso'], int(row['volume'])] for _, row in daily.iterrows()]
    return {
        'title': 'HITL rejection rate and volume over time',
        'x_axis_label': 'Date',
        'y_axis_label': 'Rejection Rate / Volume',
        'data': {
            'Rejection Rate': data1,
            'Volume': data2
        }
    }


def _compute_reviewer_time_series(df: pd.DataFrame):
    """
    Per-reviewer daily rejection rate and volume (reviewer vs self over time).
    Returns dict: { "reviewer_id": { "dates": [iso_date, ...], "rejection_rate": [...], "volume": [...] }, ... }.
    """
    if df is None or df.empty:
        return {}
    if 'hitl_reviewer_id' not in df.columns or 'hitl_qa_decision' not in df.columns:
        return {}
    date_col = None
    for col in ['hitl_review_time', 'ai_verification_time', 'last_activity_timestamp', 'first_activity_timestamp']:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        return {}
    df = df.copy()
    df['hitl_reviewer_id'] = df['hitl_reviewer_id'].fillna('Unknown').astype(str)
    df['_dt'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['_dt'])
    if df.empty:
        return {}
    df['_date'] = df['_dt'].dt.date
    if not pd.api.types.is_numeric_dtype(df['hitl_qa_decision']):
        df['hitl_qa_decision'] = df['hitl_qa_decision'].apply(
            lambda x: 1.0 if str(x).strip().upper() in ("REJECTED", "REPROCESS", "PENDING") else 0.0
        )
    out = {}
    for reviewer_id, grp in df.groupby('hitl_reviewer_id'):
        daily = grp.groupby('_date', as_index=False).agg(
            rejection_rate=('hitl_qa_decision', 'mean'),
            volume=('hitl_qa_decision', 'count')
        ).sort_values('_date')
        dates = daily['_date'].astype(str).tolist()
        out[str(reviewer_id)] = {
            'dates': _to_native(dates),
            'rejection_rate': _to_native([round(x, 4) for x in daily['rejection_rate'].tolist()]),
            'volume': _to_native(daily['volume'].astype(int).tolist())
        }
    return out


def _build_comment_samples(df_sample: pd.DataFrame, max_rows: int = 100):
    """
    Build qa_feedback_samples from comparator: rows with non-empty activity_comment_sample or feedback_text_sample.
    Returns list of { businessKey, batchId, hitl_reviewer_id, activity_comment_sample, feedback_text_sample }.
    """
    if df_sample is None or df_sample.empty:
        return []
    cols = []
    for c in ('activity_comment_sample', 'feedback_text_sample'):
        if c in df_sample.columns:
            cols.append(c)
    if not cols:
        return []
    has_text = df_sample[cols[0]].fillna('').astype(str).str.strip() != ''
    for c in cols[1:]:
        has_text = has_text | (df_sample[c].fillna('').astype(str).str.strip() != '')
    subset = df_sample.loc[has_text]
    if subset.empty:
        return []
    subset = subset.tail(max_rows)
    keys = ['businessKey', 'batchId', 'hitl_reviewer_id'] if 'businessKey' in df_sample.columns else ['batchId', 'hitl_reviewer_id']
    keys = [k for k in keys if k in df_sample.columns]
    sample_cols = [c for c in ('activity_comment_sample', 'feedback_text_sample') if c in df_sample.columns]
    rows = []
    for _, r in subset.iterrows():
        obj = {}
        for k in keys:
            obj[k] = _to_native(r[k])
        for c in sample_cols:
            val = r.get(c)
            obj[c] = _to_native(val) if pd.notna(val) else ''
        rows.append(obj)
    return rows


# Monitor Output Structure: only these keys are yielded and written to test_results.json
M3_ALLOWED_KEYS = (
    'generic_table', 'generic_bar_graph', 'horizontal_bar_graph',
    'generic_scatter_plot', 'generic_donut_chart', 'generic_pie_chart',
    'time_line_graph'
)
M3_TOP_N_FEATURES = 20


def _build_m3_visualizations(result: dict, df_sample: pd.DataFrame) -> dict:
    """Build ModelOp chart/table payloads per Monitor Output Structure (HITL); data1/data2; reviewer/QA in generic_table only."""
    out = {}
    reviewer_stats = _compute_reviewer_stats(df_sample)
    time_series = _compute_time_series(df_sample)
    comment_samples = _build_comment_samples(df_sample)
    categories, psi_vals, js_vals = [], [], []
    stab = result.get('stability')
    if stab and isinstance(stab, list) and len(stab) > 0 and 'values' in stab[0]:
        vals = stab[0]['values']
        for fname in vals:
            v = vals[fname]
            if isinstance(v, dict) and 'stability_index' in v:
                categories.append(fname)
                psi_vals.append(_to_native(v['stability_index']))
                js_vals.append(_to_native(result.get(fname + '_js_distance')) if result.get(fname + '_js_distance') is not None else 0)
    if not categories:
        for k in result:
            if isinstance(k, str) and k.endswith('_CSI'):
                feat = k.replace('_CSI', '')
                categories.append(feat)
                psi_vals.append(_to_native(result[k]) if result.get(k) is not None else 0)
                jv = result.get(feat + '_js_distance')
                js_vals.append(_to_native(jv) if jv is not None else 0)
    n_full = len(categories)
    n = min(n_full, M3_TOP_N_FEATURES) if n_full else 0
    if n:
        categories = categories[:n]
        psi_list = psi_vals[:n]
        js_list = (js_vals + [0] * n)[:n]
        pretty_categories = [_pretty_feature_name(c) for c in categories]
        out['generic_bar_graph'] = {
            'title': 'HITL Reviewer Calibration and Drift by Key Features',
            'x_axis_label': 'Monitored Feature',
            'y_axis_label': 'CSI / JS Value',
            'rotated': False,
            'data': {'data1': psi_list, 'data2': js_list},
            'categories': pretty_categories
        }
        out['horizontal_bar_graph'] = {
            'title': 'HITL Reviewer Calibration and Drift by Key Features (Horizontal)',
            'x_axis_label': 'CSI / JS Value',
            'y_axis_label': 'Monitored Feature',
            'rotated': True,
            'data': {'data1': psi_list, 'data2': js_list},
            'categories': pretty_categories
        }
        scatter_pts = [[psi_list[i], js_list[i]] for i in range(n) if psi_list[i] is not None and js_list[i] is not None]
        if scatter_pts:
            out['generic_scatter_plot'] = {
                'title': 'HITL Feature Drift Relationship (CSI vs JS Distance)',
                'x_axis_label': 'CSI (Stability Index)',
                'y_axis_label': 'Jensen–Shannon distance',
                'type': 'scatter',
                'data': {'data1': scatter_pts}
            }
    rows = []
    if 'CSI_maxCSIValue' in result:
        rows.append({'Metric': 'Largest Stability Shift (CSI)', 'Feature': _pretty_feature_name(result.get('CSI_maxCSIValueFeature', '')), 'Value': _to_native(result['CSI_maxCSIValue'])})
    if 'CSI_minCSIValue' in result:
        rows.append({'Metric': 'Smallest Stability Shift (CSI)', 'Feature': _pretty_feature_name(result.get('CSI_minCSIValueFeature', '')), 'Value': _to_native(result['CSI_minCSIValue'])})
    score_psi_key = next((k for k in result if k.endswith('_PSI')), None)
    if score_psi_key:
        rows.append({'Metric': 'Overall Prediction Shift (PSI)', 'Feature': _pretty_feature_name(score_psi_key.replace('_PSI', '')), 'Value': _to_native(result[score_psi_key])})
    if not rows:
        rows.append({'Metric': 'Stability/Drift', 'Feature': '-', 'Value': '-'})
    if reviewer_stats:
        total_vol = sum(r['Volume'] for r in reviewer_stats)
        if 'hitl_qa_decision' in df_sample.columns and pd.api.types.is_numeric_dtype(df_sample['hitl_qa_decision']):
            team_avg = float(df_sample['hitl_qa_decision'].mean())
        else:
            team_avg = (sum(r['Rejection Rate'] * r['Volume'] for r in reviewer_stats) / total_vol) if total_vol else None
        if team_avg is not None:
            rows.append({'Metric': 'Team Avg Rejection Rate', 'Feature': 'Reviewer Analysis', 'Value': _to_native(round(team_avg, 4))})
        for r in reviewer_stats:
            rows.append({'Metric': f"Reviewer {r['Reviewer']} Rejection Rate", 'Feature': 'Reviewer Analysis', 'Value': r['Rejection Rate']})
            rows.append({'Metric': f"Reviewer {r['Reviewer']} vs Team", 'Feature': 'Reviewer Analysis', 'Value': r['vs Team']})
    if comment_samples:
        rows.append({'Metric': 'QA feedback samples count', 'Feature': 'HITL', 'Value': len(comment_samples)})
    out['generic_table'] = rows
    if time_series:
        ts = time_series
        data = ts.get('data', {})
        out['time_line_graph'] = {
            'title': ts.get('title', 'Daily HITL Rejection Rate and Review Volume'),
            'x_axis_label': ts.get('x_axis_label', 'Date'),
            'y_axis_label': ts.get('y_axis_label', 'Rejection Rate / Volume'),
            'data': {'data1': data.get('Rejection Rate', []), 'data2': data.get('Volume', [])}
        }
    if 'hitl_qa_decision' in df_sample.columns:
        vc = df_sample['hitl_qa_decision'].astype(str).value_counts()
        counts = _to_native(vc.tolist())
        cats = _to_native(vc.index.tolist())
        out['generic_donut_chart'] = {
            'title': 'Comparator HITL Decision Mix',
            'type': 'donut',
            'data': {'data1': counts},
            'categories': cats
        }
        out['generic_pie_chart'] = {
            'title': 'Comparator HITL Decision Mix',
            'type': 'pie',
            'data': {'data1': counts},
            'categories': cats
        }
    return out


# modelop.init
def init(job_json: dict) -> None:
    """
    Initializes the job, extracts group information, and validates schema fail-fast.
    
    Args:
        job_json (dict): job JSON
    """
    global JOB
    global GROUP
    
    # Extract job_json and validate schema using the attached UI asset
    JOB = job_json
    infer.validate_schema(job_json)
    
    # Extract GROUP specifically for stability analysis
    # Uses rawJson to safely traverse the underlying model representation
    try:
        job = json.loads(job_json.get("rawJson", "{}"))
        GROUP = job.get('referenceModel', {}).get('group', None)
    except Exception as e:
        logger.warning(f"Could not extract GROUP from rawJson: {e}")
        GROUP = None

# modelop.metrics
def metrics(df_baseline: pd.DataFrame, df_sample: pd.DataFrame) -> dict:
    """
    Computes stability and drift for HITL. Yields only Monitor Output Structure keys:
    generic_table, generic_bar_graph, horizontal_bar_graph, generic_scatter_plot,
    generic_donut_chart, generic_pie_chart, time_line_graph. Reviewer/QA summary and
    dates are in generic_table rows.

    Weight variable: Input data must include a numeric column "weight" (default 1.0 from
    preprocess). Only this numeric column is used as weight. You can later add business
    logic so weight varies by record. Example: weight = 2.0 when (comments indicate
    "not aligned with company policy" or a flag like "high_risk_batch" is true);
    otherwise 1.0. Implement by setting the weight column in the preprocess or upstream
    so the dataframe passed here already has the desired per-record weights.
    """
    # Ensure score column is numeric for stability (weight * score); avoid ufunc 'divide' errors
    df_baseline = df_baseline.copy()
    df_sample = df_sample.copy()
    if "ai_overall_status" in df_baseline.columns and not pd.api.types.is_numeric_dtype(df_baseline["ai_overall_status"]):
        for df in (df_baseline, df_sample):
            if "ai_overall_status" in df.columns:
                df["ai_overall_status"] = df["ai_overall_status"].apply(
                    lambda x: 1.0 if str(x).strip().upper() == "FAIL" else 0.0
                )
    if "hitl_qa_decision" in df_baseline.columns and not pd.api.types.is_numeric_dtype(df_baseline["hitl_qa_decision"]):
        for df in (df_baseline, df_sample):
            if "hitl_qa_decision" in df.columns:
                df["hitl_qa_decision"] = df["hitl_qa_decision"].apply(
                    lambda x: 1.0 if str(x).strip().upper() in ("REJECTED", "REPROCESS", "PENDING") else 0.0
                )

    # 1. Initialize & Compute Stability Metrics (PSI, CSI)
    stability_monitor = stability.StabilityMonitor(
        df_baseline=df_baseline,
        df_sample=df_sample,
        job_json=JOB
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

    # 3. Concatenate result
    result = utils.merge(
        stability_metrics,
        es_drift,
        js_drift,
        kl_drift,
        ks_drift,
        summary_drift
    )
    viz = _build_m3_visualizations(result, df_sample)
    baseline_first, baseline_last = _get_date_range(df_baseline)
    sample_first, sample_last = _get_date_range(df_sample)
    if viz.get('generic_table') is not None:
        if sample_first is not None:
            viz['generic_table'].append({'Metric': 'First prediction date', 'Feature': 'Comparator', 'Value': sample_first})
        if sample_last is not None:
            viz['generic_table'].append({'Metric': 'Last prediction date', 'Feature': 'Comparator', 'Value': sample_last})
        if baseline_first is not None:
            viz['generic_table'].append({'Metric': 'Baseline first date', 'Feature': 'Baseline', 'Value': baseline_first})
        if baseline_last is not None:
            viz['generic_table'].append({'Metric': 'Baseline last date', 'Feature': 'Baseline', 'Value': baseline_last})
    output = {k: viz[k] for k in M3_ALLOWED_KEYS if k in viz}
    yield output

if __name__ == "__main__":
    # Local Testing Execution Block (Slide 38 ModelOp Developer Training)
    # Assumes mtr_preprocess.py has generated the files in the current directory.
    
    print("Testing Monitor 3 locally...")
    
    # 1. Load the mock job JSON to simulate the platform environment
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(script_dir, 'modelop_schema.json'), 'r') as f:
            schema_from_file = json.load(f)
    except FileNotFoundError:
        print("[!] modelop_schema.json not found. Run mtr_preprocess.py first.")
        sys.exit(1)

    # Convert preprocess schema (inputSchema.items.properties) to ModelOp infer format (fields array)
    props = schema_from_file.get("inputSchema", {}).get("items", {}).get("properties", {})
    fields = []
    score_seen = False
    for name, p in props.items():
        role = p.get("role", "predictor")
        if role == "score":
            if score_seen:
                role = "predictor"
            else:
                score_seen = True
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
        df_b = pd.read_json(os.path.join(script_dir, 'CHIP_mtr_3_baseline.json'), orient='records')
        df_c = pd.read_json(os.path.join(script_dir, 'CHIP_mtr_3_comparator.json'), orient='records')
        if df_b.empty and df_c.empty:
            print("[!] Both baseline and comparator are empty. Run preprocess with data that yields both splits.")
            sys.exit(1)
        if df_b.empty and len(df_c) > 0:
            n = max(1, len(df_c) // 2)
            df_b = df_c.iloc[:n].copy()
            df_c = df_c.iloc[n:].copy()
            print("[*] Baseline was empty; using first half of comparator as baseline for local test.")
    except Exception as e:
         print(f"[!] Error loading test data: {e}")
         sys.exit(1)
         
    # 4. Call metrics() and 5. Always write metrics payload to JSON (even on error, so a file is produced every run)
    out_path = os.path.join(script_dir, 'CHIP_mtr_3_test_results.json')

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
        results = list(metrics(df_b, df_c))
        payload = _nan_to_none(results[0])
    except Exception as e:
        payload = {"error": str(e), "metrics_computed": False}
        print(f"[!] metrics() failed: {e}")

    wrapped_payload = [payload]
    with open(out_path, 'w') as f:
        json.dump(wrapped_payload, f, indent=2, default=_json_serial)
    print(f"\n[SUCCESS] Output written to {out_path}")
    print(json.dumps(wrapped_payload, indent=2, default=_json_serial))