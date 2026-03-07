"""OOTB ModelOp Center model to get a statistical summary of an asset"""

import pandas

import modelop.monitors.volumetrics as volumetrics
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
    """A function to get a summary (min, max, standard deviation, etc.) of an asset

    Args:
        dataframe (pandas.DataFrame): Input dataframe

    Returns:
        dict: Statistical summary
    """

    # Initialize Volumetric monitor with 1st input DataFrame
    volumetric_monitor = volumetrics.VolumetricMonitor(dataframe=dataframe)

    yield volumetric_monitor.summary(job_json=JOB)
