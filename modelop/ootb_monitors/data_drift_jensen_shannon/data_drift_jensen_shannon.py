"""OOTB ModelOp Center model to compute Jensen-Shannon distances on input data"""

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
        job_json (dict[str,str]): job JSON in a string format.
    """

    # Extract job JSON
    global JOB
    JOB = job_json
    infer.validate_schema(job_json)


# modelop.metrics
def metrics(df_baseline: pandas.DataFrame, df_sample: pandas.DataFrame) -> dict:
    """A function to compute Jensen-Shannon distances given baseline and sample (prod) datasets.

    Args:
        df_baseline (pandas.DataFrame): Baseline dataset containing model inputs
        df_sample (pandas.DataFrame): Sample (prod) dataset containing model inputs

    Returns:
        dict: Data drift metrics (JS distance(s))
    """

    # Initialize DriftDetector
    drift_detector = drift.DriftDetector(
        df_baseline=df_baseline, df_sample=df_sample, job_json=JOB
    )

    # Compute drift metrics
    yield drift_detector.calculate_drift(
        pre_defined_test="Jensen-Shannon", flattening_suffix="_js_distance"
    )
