"""OOTB ModelOp Center model to compute stability metrics (PSI, CSI)"""
import pandas

import modelop.monitors.stability as stability
import modelop.schema.infer as infer
import modelop.utils as utils
import json

DEPLOYABLE_MODEL = {}

logger = utils.configure_logger()

JOB = {}

# modelop.init
def init(job_json: dict) -> None:
    """A function to receive the job JSON and validate schema fail-fast.

    Args:
        job_json (dict): job JSON
    """

    # Extract job_json and validate schema
    global JOB
    global DEPLOYABLE_MODEL
    JOB = job_json
    infer.validate_schema(job_json)

    job = json.loads(job_json["rawJson"])
    DEPLOYABLE_MODEL = job["referenceModel"]


# modelop.metrics
def metrics(df_baseline: pandas.DataFrame, df_sample: pandas.DataFrame) -> dict:
    """A function to compute stability indices given baseline and sample datasets

    Args:
        df_baseline (pandas.DataFrame): Baseline dataset containing model inputs and score (output)
        df_sample (pandas.DataFrame): Sample (prod) dataset containing model inputs and score

    Returns:
        dict: Stability metrics for each input feature and score column
    """

    # Initialize StabilityMonitor
    stability_monitor = stability.StabilityMonitor(
        df_baseline=df_baseline,
        df_sample=df_sample,
        job_json=JOB,
    )

    # Compute stability metrics
    result = stability_monitor.compute_stability_indices()
    result.update({
        "modelUseCategory": DEPLOYABLE_MODEL.get("storedModel", {})
        .get("modelMetaData", {})
        .get("modelUseCategory", ""),
        "modelOrganization": DEPLOYABLE_MODEL.get("storedModel", {})
        .get("modelMetaData", {})
        .get("modelOrganization", ""),
        "modelRisk": DEPLOYABLE_MODEL.get("storedModel", {})
        .get("modelMetaData", {})
        .get("modelRisk", ""),
        "modelMethodology": DEPLOYABLE_MODEL.get("storedModel", {})
        .get("modelMetaData", {})
        .get("modelMethodology", "")
    })
    yield result
