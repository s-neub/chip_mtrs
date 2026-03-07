"""OOTB ModelOp Center model to return the record count of an asset"""
import pandas

import modelop.monitors.volumetrics as volumetrics
from modelop.schema import infer
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
    """A function to return the record count of an asset

    Args:
        dataframe (pandas.DataFrame): Input dataframe

    Returns:
        dict: Record count of input asset
    """

    # Initialize Volumetric monitor with 1st input DataFrame
    volumetric_monitor = volumetrics.VolumetricMonitor(dataframe=dataframe)

    yield volumetric_monitor.count(job_json=JOB)
