"""OOTB ModelOp Center model to compute stability metrics (PSI, CSI)"""
import pandas
import json
import modelop.monitors.stability as stability
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

    # Extract job_json and validate schema
    global JOB
    JOB = job_json
    infer.validate_schema(job_json)
    global GROUP
    job = json.loads(job_json["rawJson"]) 
    GROUP=job.get('referenceModel', {}).get('group', None) 



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
    results=stability_monitor.compute_stability_indices()
    results.update({"group":GROUP})
    yield results
