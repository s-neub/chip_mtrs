"""OOTB ModelOp Center model to compute Box tidwell to monitor linearity"""

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
    """A function to compute Box-Tidwell p-values on sample data

    Args:
        dataframe (pandas.DataFrame): Sample prod data containing labels (ground truths)
        and numerical_columns (predictors)

    Returns:
        (dict): Box-Tidwell results
    """

    # Initialize metrics class
    linearity_metrics = diagnostics.LinearityMetrics(dataframe=dataframe, job_json=JOB)

    # Run test
    yield linearity_metrics.box_tidwell()
