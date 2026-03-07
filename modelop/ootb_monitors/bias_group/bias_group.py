"""OOTB ModelOp Center model to compute group metrics on protected classes"""

import pandas

import modelop.monitors.bias as bias
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

    # Extract job_json and validate
    global JOB
    JOB = job_json
    infer.validate_schema(job_json)


# modelop.metrics
def metrics(dataframe: pandas.DataFrame) -> dict:
    """A function to compute group metrics on sample prod data

    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs),
        labels (ground truths) and protected classes (e.g. "gender", "race", etc.)

    Raises:
        ValueError: If schema contains no protected classes

    Returns:
        (dict): Group metrics for each protected class
    """

    bias_monitor = bias.BiasMonitor(dataframe=dataframe, job_json=JOB)

    yield bias_monitor.compute_group_metrics(pre_defined_test="aequitas_group")
