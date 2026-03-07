"""OOTB ModelOp Center model to detect discrepancies between 2 assets based on their identifiers"""

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
def metrics(df_1: pandas.DataFrame, df_2: pandas.DataFrame) -> dict:
    """A fucntion to detect discrepancies between two assets based on their record identifiers"

    Args:
        df_1 (pandas.DataFrame): First dataframe
        df_2 (pandas.DataFrame): Second dataframe

    Returns:
        dict: A breakdown of mismatches between input dataframe, based on identifier columns
    """

    # Initialize Volumetric monitor with 1st input DataFrame
    volumetric_monitor = volumetrics.VolumetricMonitor(dataframe=df_1)

    # Compare DataFrames on identifier_columns
    yield volumetric_monitor.identifier_comparison(dataframe_2=df_2, job_json=JOB)
