"""OOTB ModelOp Center model to compute Variance Inflation Factors to monitor multicollinearity"""

import pandas

import modelop.schema.infer as infer
import modelop.stats.diagnostics as diagnostics
import modelop.utils as utils

logger = utils.configure_logger()

JOB = {}

# modelop.init
def init(job_json: dict) -> None:
    """A function to extract input schema from job JSON and set monitoring parameters

    Args:
        job_json (dict): job JSON
    """

    # Extract job_json and validate schema
    global JOB
    JOB = job_json
    infer.validate_schema(job_json)


# modelop.metrics
def metrics(dataframe: pandas.DataFrame) -> dict:
    """A function to compute Variance Inflation Factors on sample data

    Args:
        dataframe (pandas.DataFrame): Sample prod data containing numerical_columns (predictors)

    Returns:
        (dict): Pearson Correlation results
    """

    # Initialize metrics class
    multicollinearity_metrics = diagnostics.MulticollinearityMetrics(
        dataframe=dataframe, job_json=JOB
    )

    # Run test
    yield multicollinearity_metrics.variance_inflation_factor()
