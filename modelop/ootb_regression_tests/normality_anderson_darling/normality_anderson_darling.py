"""OOTB ModelOp Center model to run the Anderson-Darling test to monitor normality"""

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
    """A function to run the Breauch-Pagan test on sample data

    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs) and
        labels (ground truths)

    Returns:
        (dict): Anderson-Darling test results
    """

    # Initialize metrics class
    normality_metrics = diagnostics.NormalityMetrics(dataframe=dataframe, job_json=JOB)

    # Run test
    yield normality_metrics.anderson_darling_test()
