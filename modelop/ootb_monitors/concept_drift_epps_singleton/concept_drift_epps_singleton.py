"""OOTB ModelOp Center model to compute the Epps-Singleton p-value on output data"""

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
    """A function to compute the Epps-Singleton p-value given baseline and sample (prod) datasets.

    Args:
        df_baseline (pandas.DataFrame): Baseline dataset containing model output (score)
        df_sample (pandas.DataFrame): Sample (prod) dataset containing model output (score)

    Returns:
        dict: Concept drift metric (ES p-value)
    """

    # Initialize DriftDetector
    concept_drift_detector = drift.ConceptDriftDetector(
        df_baseline=df_baseline, df_sample=df_sample, job_json=JOB
    )

    # Compute concept drift metrics
    yield concept_drift_detector.calculate_concept_drift(
        pre_defined_test="Epps-Singleton", flattening_suffix="_es_pvalue"
    )
