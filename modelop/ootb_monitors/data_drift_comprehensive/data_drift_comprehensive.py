"""OOTB ModelOp Center model to run KS, ES, JS, KL, and summary drift methods on input data"""

import pandas

import modelop.monitors.drift as drift
import modelop.schema.infer as infer
import modelop.utils as utils

logger = utils.configure_logger()

JOB = {}

# modelop.init
def init(job_json: dict) -> None:
    """A function to receive the job JSON and validate schema fail-fast.

    Args:
        job_json (dict): job JSON
    """

    # Extract job JSON
    global JOB
    JOB = job_json
    infer.validate_schema(job_json)


# modelop.metrics
def metrics(df_baseline: pandas.DataFrame, df_sample: pandas.DataFrame) -> dict:
    """A function to compute data drift metrics given baseline and sample (prod) datasets

    Args:
        df_baseline (pandas.DataFrame): Baseline dataset containing model inputs
        df_sample (pandas.DataFrame): Sample (prod) dataset containing model inputs

    Returns:
        dict: Data drift metrics (ES, JS, KL, KS, Summary)
    """

    # Initialize DriftDetector
    drift_detector = drift.DriftDetector(
        df_baseline=df_baseline, df_sample=df_sample, job_json=JOB
    )

    # Compute drift metrics
    # Epps-Singleton p-values
    es_drift_metrics = drift_detector.calculate_drift(
        pre_defined_test="Epps-Singleton", flattening_suffix="_es_pvalue"
    )

    # Jensen-Shannon distance
    js_drift_metrics = drift_detector.calculate_drift(
        pre_defined_test="Jensen-Shannon", flattening_suffix="_js_distance"
    )

    # Kullback-Leibler divergence
    kl_drift_metrics = drift_detector.calculate_drift(
        pre_defined_test="Kullback-Leibler",
        flattening_suffix="_kl_divergence",
    )

    # Kolmogorov-Smirnov p-values
    ks_drift_metrics = drift_detector.calculate_drift(
        pre_defined_test="Kolmogorov-Smirnov", flattening_suffix="_ks_pvalue"
    )

    # Pandas summary
    summary_drift_metrics = drift_detector.calculate_drift(pre_defined_test="Summary")

    result = utils.merge(
        es_drift_metrics,
        js_drift_metrics,
        kl_drift_metrics,
        ks_drift_metrics,
        summary_drift_metrics,
    )

    yield result