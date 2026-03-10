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


def _build_m3_visualizations(result: dict, df_sample: pd.DataFrame) -> dict:
    """Build ModelOp chart/table/donut/pie payloads per Monitor Output Structure (HITL)."""
    out = {}
    reviewer_stats = _compute_reviewer_stats(df_sample)
    time_series = _compute_time_series(df_sample)
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
    n = len(categories)
    if n:
        psi_list = psi_vals[:n]
        js_list = (js_vals + [0] * n)[:n]
        out['generic_bar_graph'] = {
            'title': 'Stability / Drift by feature (HITL)',
            'x_axis_label': 'Feature',
            'y_axis_label': 'Index / Distance',
            'rotated': False,
            'data': {'PSI': psi_list, 'JS Distance': js_list},
            'categories': categories
        }
        out['horizontal_bar_graph'] = {
            'title': 'Stability / Drift by feature (HITL, horizontal)',
            'x_axis_label': 'Index / Distance',
            'y_axis_label': 'Feature',
            'rotated': True,
            'data': {'PSI': psi_list, 'JS Distance': js_list},
            'categories': categories
        }
        scatter_pts = [[psi_list[i], js_list[i]] for i in range(n) if psi_list[i] is not None and js_list[i] is not None]
        if scatter_pts:
            out['generic_scatter_plot'] = {
                'title': 'Stability (CSI) vs Drift (JS) by feature (HITL)',
                'x_axis_label': 'CSI (Stability Index)',
                'y_axis_label': 'Jensen–Shannon distance',
                'type': 'scatter',
                'data': {'Features': scatter_pts}
            }
    rows = []
    if 'CSI_maxCSIValue' in result:
        rows.append({'Metric': 'Max CSI', 'Feature': result.get('CSI_maxCSIValueFeature', ''), 'Value': _to_native(result['CSI_maxCSIValue'])})
    if 'CSI_minCSIValue' in result:
        rows.append({'Metric': 'Min CSI', 'Feature': result.get('CSI_minCSIValueFeature', ''), 'Value': _to_native(result['CSI_minCSIValue'])})
    score_psi_key = next((k for k in result if k.endswith('_PSI')), None)
    if score_psi_key:
        rows.append({'Metric': 'Score PSI', 'Feature': score_psi_key.replace('_PSI', ''), 'Value': _to_native(result[score_psi_key])})
    if not rows:
        rows.append({'Metric': 'Stability/Drift', 'Feature': '-', 'Value': '-'})
    if reviewer_stats:
        out['reviewer_stats_table'] = _to_native(reviewer_stats)
        if 'hitl_qa_decision' in df_sample.columns and pd.api.types.is_numeric_dtype(df_sample['hitl_qa_decision']):
            team_avg = float(df_sample['hitl_qa_decision'].mean())
        else:
            total_v = sum(r['Volume'] for r in reviewer_stats)
            team_avg = (sum(r['Rejection Rate'] * r['Volume'] for r in reviewer_stats) / total_v) if total_v else None
        if team_avg is not None:
            rows.append({'Metric': 'Team Avg Rejection Rate', 'Feature': 'Reviewer Analysis', 'Value': _to_native(round(team_avg, 4))})
        for r in reviewer_stats:
            rows.append({'Metric': f"Reviewer {r['Reviewer']} Rejection Rate", 'Feature': 'Reviewer Analysis', 'Value': r['Rejection Rate']})
            rows.append({'Metric': f"Reviewer {r['Reviewer']} vs Team", 'Feature': 'Reviewer Analysis', 'Value': r['vs Team']})
    if time_series:
        out['time_line_graph'] = _to_native(time_series)
    out['generic_table'] = rows
    if 'hitl_qa_decision' in df_sample.columns:
        vc = df_sample['hitl_qa_decision'].astype(str).value_counts()
        counts = _to_native(vc.tolist())
        cats = _to_native(vc.index.tolist())
        out['generic_donut_chart'] = {
            'title': 'HITL QA decision (comparator)',
            'type': 'donut',
            'data': {'Count': counts},
            'categories': cats
        }
        out['generic_pie_chart'] = {
            'title': 'HITL QA decision (comparator)',
            'type': 'pie',
            'data': {'Count': counts},
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
    Computes combined stability and data drift metrics for Human inputs.

    Output payload structure (all keys are emitted so the ModelOp UI can display
    every element; new users can see what monitor outputs are available):
    - stability: list of stability analysis results (PSI/CSI per feature).
    - data_drift: list of drift test results.
    - CSI_maxCSIValue, CSI_maxCSIValueFeature, CSI_minCSIValue, CSI_minCSIValueFeature.
    - <feature>_CSI, <score_column>_PSI, <feature>_js_distance.
    - generic_bar_graph, horizontal_bar_graph: PSI/Drift by feature (vertical and horizontal).
    - generic_scatter_plot: CSI vs JS distance by feature.
    - generic_table: list of {"Metric", "Feature", "Value"} rows.
    - generic_donut_chart, generic_pie_chart: part-to-whole (HITL QA decision comparator).
    - firstPredictionDate, lastPredictionDate: comparator date range (ISO) for tracking monitor outputs over time.
    - baseline_firstDate, baseline_lastDate: baseline date range (ISO); timestamp data associated with baseline/comparator.

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
    # 4. Add ModelOp chart/table payloads for UI
    viz = _build_m3_visualizations(result, df_sample)
    result.update(viz)
    # 5. Associate timestamp range with baseline/comparator for tracking over time (ModelOp firstPredictionDate/lastPredictionDate)
    baseline_first, baseline_last = _get_date_range(df_baseline)
    sample_first, sample_last = _get_date_range(df_sample)
    result['baseline_firstDate'] = baseline_first
    result['baseline_lastDate'] = baseline_last
    result['firstPredictionDate'] = sample_first
    result['lastPredictionDate'] = sample_last
    yield result

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

    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, default=_json_serial)
    print(f"\n[SUCCESS] Output written to {out_path}")
    print(json.dumps(payload, indent=2, default=_json_serial))