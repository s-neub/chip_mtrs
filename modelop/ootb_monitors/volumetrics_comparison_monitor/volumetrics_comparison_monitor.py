"""OOTB ModelOp Center model to compare two datasets based on their record counts & identifiers"""
import pandas

import modelop.monitors.volumetrics as volumetrics
import modelop.schema.infer as infer
import modelop.utils as utils

logger = utils.configure_logger()

JOb = {}

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
    """A function to compare two datasets on their record counts & identifiers

    Args:
        df_1 (pandas.DataFrame): First DataFrame
        df_2 (pandas.DataFrame): Second DataFrame

    Returns:
        dict: Count difference, identifiers match? (boolean), and identifier comparison
    """

    # Initialize Volumetric monitor with 1st input DataFrame
    volumetric_monitor = volumetrics.VolumetricMonitor(dataframe=df_1)

    # Compare DataFrames on identifier_columns
    identifiers_comparison = volumetric_monitor.identifier_comparison(
        df_2, job_json=JOB
    )

    count_comparison = volumetric_monitor.count_comparison(dataframe_2=df_2)

    yield utils.merge(identifiers_comparison, count_comparison)
