"""OOTB ModelOp Center model to detect discrepancies between two assets based on record counts"""

import pandas

import modelop.monitors.volumetrics as volumetrics
import modelop.utils as utils

logger = utils.configure_logger()


# modelop.metrics
def metrics(df_1: pandas.DataFrame, df_2: pandas.DataFrame) -> dict:
    """A function to detect discrepancies between two assets based on their record counts

    Args:
        df_1 (pandas.DataFrame): First dataframe
        df_2 (pandas.DataFrame): Second dataframe

    Returns:
        dict: Record counts and their difference
    """

    # Initialize Volumetric monitor with 1st input DataFrame
    volumetric_monitor = volumetrics.VolumetricMonitor(dataframe=df_1)

    yield volumetric_monitor.count_comparison(dataframe_2=df_2)
