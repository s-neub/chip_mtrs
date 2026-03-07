"""OOTB ModelOp Center model to compute binary classification metrics"""
import pandas

import modelop.monitors.performance as performance
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
    """A function to compute binary classification metrics given a sample (prod) dataset

    Args:
        dataframe (pandas.DataFrame): Sample (prod) dataset containing scores (model outputs)
        and labels (ground truths)

    Returns:
        dict: Binary classification metrics (accuracy, precision, recall, AUC, F1_score, confusion
        matrix)
    """

    # Initialize ModelEvaluator
    model_evaluator = performance.ModelEvaluator(dataframe=dataframe, job_json=JOB)

    # Compute classification metrics
    yield model_evaluator.evaluate_performance(
        pre_defined_metrics="classification_metrics"
    )
