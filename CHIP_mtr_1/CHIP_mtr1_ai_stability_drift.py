"""
ModelOp Center Monitor 1: AI Output Stability (PSI) & Data Drift
----------------------------------------------------------------
Tracks the behavior and drift of the Claude AI model's outputs.
Merges OOTB Stability Analysis and Comprehensive Data Drift.

Best Practice: keep init lightweight and rely on runtime-provided dataframes.
"""

import os
import pandas as pd
import json
import modelop.monitors.stability as stability
import modelop.monitors.drift as drift
import modelop.utils as utils
import sys

logger = utils.configure_logger()

JOB = {}
GROUP = None
JOB_PARAMETERS = {}
M1_AI_FAIL_VALUES = {"FAIL"}


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


# Monitor Output Structure keys for Monitor 1
M1_TABLE_KEY = 'ai_stability_drift_summary_table'
M1_BAR_KEY = 'ai_stability_drift_bar_graph'
M1_HBAR_KEY = 'ai_stability_drift_horizontal_bar_graph'
M1_SCATTER_KEY = 'ai_stability_drift_scatter_plot'
M1_DONUT_KEY = 'ai_outcome_mix_donut_chart'
M1_PIE_KEY = 'ai_outcome_mix_pie_chart'
M1_ALLOWED_KEYS = (
    M1_TABLE_KEY, M1_BAR_KEY, M1_HBAR_KEY,
    M1_SCATTER_KEY, M1_DONUT_KEY, M1_PIE_KEY
)
# Limit bar/scatter to top N features for aggregate summary
M1_TOP_N_FEATURES = 20


def _normalized_value_set(raw_values, default_values):
    """Normalize job-parameter values into a case-insensitive string set."""
    if raw_values is None:
        return {str(v).strip().upper() for v in default_values}
    if isinstance(raw_values, str):
        values = [v.strip() for v in raw_values.split(",") if v.strip()]
    elif isinstance(raw_values, (list, tuple, set)):
        values = [str(v).strip() for v in raw_values if str(v).strip()]
    else:
        values = [str(raw_values).strip()]
    if not values:
        values = [str(v).strip() for v in default_values]
    return {v.upper() for v in values}


def _build_m1_visualizations(result: dict, df_sample: pd.DataFrame) -> dict:
    """Build ModelOp chart/table payloads per Monitor Output Structure (bar, horizontal bar, table, scatter, donut, pie)."""
    out = {}
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
    # Limit to top N for aggregate summary
    n_full = len(categories)
    n = min(n_full, M1_TOP_N_FEATURES) if n_full else 0
    if n:
        categories = categories[:n]
        psi_list = psi_vals[:n]
        js_list = (js_vals + [0] * n)[:n]
        pretty_categories = [_pretty_feature_name(c) for c in categories]
        out[M1_BAR_KEY] = {
            'title': 'AI Stability vs Drift by Key Features (Baseline vs Comparator)',
            'x_axis_label': 'Monitored Feature',
            'y_axis_label': 'CSI / JS Value',
            'rotated': False,
            'data': {
                'csi_stability_index': psi_list,
                'js_drift_distance': js_list
            },
            'categories': pretty_categories
        }
        out[M1_HBAR_KEY] = {
            'title': 'AI Stability vs Drift by Key Features (Horizontal)',
            'x_axis_label': 'CSI / JS Value',
            'y_axis_label': 'Monitored Feature',
            'rotated': True,
            'data': {
                'csi_stability_index': psi_list,
                'js_drift_distance': js_list
            },
            'categories': pretty_categories
        }
        scatter_pts = [[psi_list[i], js_list[i]] for i in range(n) if psi_list[i] is not None and js_list[i] is not None]
        if scatter_pts:
            out[M1_SCATTER_KEY] = {
                'title': 'Feature Drift Relationship (CSI vs JS Distance)',
                'x_axis_label': 'CSI (Stability Index)',
                'y_axis_label': 'Jensen–Shannon distance',
                'type': 'scatter',
                'data': {'feature_points': scatter_pts}
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
    if 'activity_comment_count' in df_sample.columns and 'feedback_text_count' in df_sample.columns:
        total_act = int(pd.to_numeric(df_sample['activity_comment_count'], errors='coerce').fillna(0).sum())
        total_fb = int(pd.to_numeric(df_sample['feedback_text_count'], errors='coerce').fillna(0).sum())
        rows.append({'Metric': 'Comparator activity comment count (total)', 'Feature': 'Activity/Feedback', 'Value': total_act})
        rows.append({'Metric': 'Comparator feedback text count (total)', 'Feature': 'Activity/Feedback', 'Value': total_fb})
    out[M1_TABLE_KEY] = rows
    if 'ai_overall_status' in df_sample.columns:
        vc = df_sample['ai_overall_status'].astype(str).value_counts()
        counts = _to_native(vc.tolist())
        cats = _to_native(vc.index.tolist())
        out[M1_DONUT_KEY] = {
            'title': 'Comparator AI Outcome Mix',
            'type': 'donut',
            'data': {'outcome_count': counts},
            'categories': cats
        }
        out[M1_PIE_KEY] = {
            'title': 'Comparator AI Outcome Mix',
            'type': 'pie',
            'data': {'outcome_count': counts},
            'categories': cats
        }
    return out

# modelop.init
def init(job_json: dict) -> None:
    """
    Initializes the job, extracts group information, and validates schema fail-fast.
    """
    global JOB
    global GROUP
    global JOB_PARAMETERS
    global M1_TOP_N_FEATURES
    global M1_AI_FAIL_VALUES

    JOB = job_json or {}

    try:
        job = json.loads(JOB.get("rawJson", "{}"))
    except Exception as e:
        logger.warning(f"Could not parse rawJson in init. Falling back to defaults: {e}")
        job = {}

    JOB_PARAMETERS = job.get("jobParameters", {}) if isinstance(job.get("jobParameters", {}), dict) else {}
    GROUP = job.get("referenceModel", {}).get("group", None)

    top_n = JOB_PARAMETERS.get("M1_TOP_N_FEATURES", M1_TOP_N_FEATURES)
    try:
        M1_TOP_N_FEATURES = max(1, int(top_n))
    except Exception:
        logger.warning(f"Invalid M1_TOP_N_FEATURES={top_n}. Using default {M1_TOP_N_FEATURES}.")

    M1_AI_FAIL_VALUES = _normalized_value_set(
        JOB_PARAMETERS.get("AI_FAIL_VALUES"),
        default_values=["FAIL"]
    )

# modelop.metrics
def metrics(df_baseline: pd.DataFrame, df_sample: pd.DataFrame) -> dict: #type: ignore
    """
    Computes combined stability and data drift metrics.

    Yields only Monitor Output Structure keys for ModelOp UI with monitor-specific
    names. Date range is included as
    generic_table rows. Raw stability/CSI/drift keys are not emitted.

    Weight variable: The input data must include a numeric column "weight" (default 1.0
    from the preprocess pipeline). Only this column is used as weight; no string or
    categorical column is used. You can later design business logic so the weight value
    changes by record. Example: set weight = 2.0 when (remarks contain "policy deviation"
    or a flag column indicates "requires_escalation"); else keep 1.0. That way records
    matching certain conditions contribute more to stability/drift metrics. Implement
    by computing weight in the preprocess or in a preprocessing step before calling
    metrics(), so the dataframe passed here already has the desired weight column.
    """
    # Ensure score column is numeric for stability (weight * score); avoid ufunc 'divide' errors
    df_baseline = df_baseline.copy()
    df_sample = df_sample.copy()
    if "ai_overall_status" in df_baseline.columns and not pd.api.types.is_numeric_dtype(df_baseline["ai_overall_status"]):
        for df in (df_baseline, df_sample):
            if "ai_overall_status" in df.columns:
                df["ai_overall_status"] = df["ai_overall_status"].apply(
                    lambda x: 1.0 if str(x).strip().upper() in M1_AI_FAIL_VALUES else 0.0
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
    # 4. Add ModelOp chart/table payloads for UI (viz only; no raw stability/CSI keys)
    viz = _build_m1_visualizations(result, df_sample)
    baseline_first, baseline_last = _get_date_range(df_baseline)
    sample_first, sample_last = _get_date_range(df_sample)
    # Add date range to generic_table for tracking (aggregate summary only)
    if viz.get(M1_TABLE_KEY) is not None:
        viz[M1_TABLE_KEY] = list(viz[M1_TABLE_KEY])
        if sample_first is not None:
            viz[M1_TABLE_KEY].append({'Metric': 'First prediction date', 'Feature': 'Comparator', 'Value': sample_first})
        if sample_last is not None:
            viz[M1_TABLE_KEY].append({'Metric': 'Last prediction date', 'Feature': 'Comparator', 'Value': sample_last})
        if baseline_first is not None:
            viz[M1_TABLE_KEY].append({'Metric': 'Baseline first date', 'Feature': 'Baseline', 'Value': baseline_first})
        if baseline_last is not None:
            viz[M1_TABLE_KEY].append({'Metric': 'Baseline last date', 'Feature': 'Baseline', 'Value': baseline_last})
    # 5. Yield only Monitor Output Structure keys (no stability, CSI_*, etc.)
    output = {k: viz[k] for k in M1_ALLOWED_KEYS if k in viz}
    yield output


if __name__ == "__main__":
    # Local Testing Execution Block (Slide 38 ModelOp Developer Training)
    # Assumes mtr_preprocess.py has generated the files in the current directory.
    
    print("Testing Monitor 1 locally...")
    
    # 1. Build a minimal mock job JSON to simulate platform init payload
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mock_job = {
        "rawJson": json.dumps({
            "referenceModel": {},
            "jobParameters": {}
        })
    }
    
    # 2. Call init()
    init(mock_job)
    
    # 3. Load test data (full columns so schema columns exist; platform may pre-filter when invoking metrics())
    try:
        df_b = pd.read_json(os.path.join(script_dir, 'CHIP_mtr_1_baseline.json'), orient='records')
        df_c = pd.read_json(os.path.join(script_dir, 'CHIP_mtr_1_comparator.json'), orient='records')
        if df_b.empty and df_c.empty:
            print("[!] Both baseline and comparator are empty. Run preprocess with data that yields both splits.")
            sys.exit(1)
        if df_b.empty and len(df_c) > 0:
            # Synthetic split so stability/drift have both non-empty: use first half as baseline, rest as comparator
            n = max(1, len(df_c) // 2)
            df_b = df_c.iloc[:n].copy()
            df_c = df_c.iloc[n:].copy()
            print("[*] Baseline was empty; using first half of comparator as baseline for local test.")
    except Exception as e:
         print(f"[!] Error loading test data: {e}")
         sys.exit(1)
         
    # 4. Call metrics() and 5. Always write metrics payload to JSON (even on error, so a file is produced every run)
    out_path = os.path.join(script_dir, 'CHIP_mtr_1_test_results.json')

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