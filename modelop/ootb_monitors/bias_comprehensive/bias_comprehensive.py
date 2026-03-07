"""OOTB ModelOp Center model to compute bias (disparity) and group metrics on protected classes"""

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
    """A function to compute group and bias (diparity) metrics on sample prod data

    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs),
        labels (ground truths) and protected classes (e.g. "gender", "race", etc.)

    Raises:
        ValueError: If schema contains no protected classes

    Returns:
        (dict): Bias (disparity) and Group metrics for each protected class
    """

    # Initialize BiasMonitor
    bias_monitor = bias.BiasMonitor(
        dataframe=dataframe,
        job_json=JOB,
    )

    # Compute aequitas_bias (disparity) metrics
    bias_metrics = bias_monitor.compute_bias_metrics(pre_defined_test="aequitas_bias")

    # Compute aequitas_group (Group) metrics
    group_metrics = bias_monitor.compute_group_metrics(
        pre_defined_test="aequitas_group"
    )

    yield utils.merge(bias_metrics, group_metrics)
