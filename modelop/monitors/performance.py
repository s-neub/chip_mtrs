"""This module provides a class to evaluate models on regression and classification metrics.

See `ModelEvaluator` for usage examples.

.. toctree::

.. autosummary::
   :toctree: _autosummary
"""
import doctest
import logging
import math
from typing import Optional, Union

# Third party packages
import numpy
import pandas
from pandas.api.types import is_numeric_dtype

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.multiclass import unique_labels

import modelop.schema.infer as infer
from modelop.monitors.assertions import (
    check_date_column,
    check_columns_in_dataframe,
    check_input_types,
    check_pre_defined_metric,
)
from modelop.utils import check_and_drop_nulls

pandas.options.mode.chained_assignment = "warn"

logger = logging.getLogger(__name__)


def type_check_and_cast(dataframe, input_schema_definition):
    filtered_fields = [v for v in input_schema_definition.get('fields', []) if (v.get('role') == 'label' or v.get('role') == 'score')]
    if len(filtered_fields) == 2:
        label = next(x for x in filtered_fields if x.get('role') == 'label')
        label_type = label.get('type')
        label_name = label.get('name')
        if label_type == 'int' and not pandas.api.types.is_integer(dataframe.get(label_name)):
            try:
                dataframe[label_name] = dataframe[label_name].astype(int)
            except Exception as ex:
                logger.warning(f'Schema specifies int type but data is not convertible to int: {str(ex)}')

        score = next(x for x in filtered_fields if x.get('role') == 'score')
        score_type = score.get('type')
        score_name = score.get('name')
        if score_type == 'int' and not pandas.api.types.is_integer(dataframe.get(score_name)):
            try:
                dataframe[score_name] = dataframe[score_name].astype(int)
            except Exception as ex:
                logger.warning(f'Schema specifies int type but data is not convertible to int: {str(ex)}')

    return dataframe


class ModelEvaluator:
    """
    A class to evaluate the performance of a ML model on a scored and labeled dataset.

    Args:
        dataframe (pandas.DataFrame): Pandas DataFrame.

        job_json (dict): JSON dictionary with the metadata of the model.

        score_column (str): Column containing predicted values (as computed by
            underlying model).

        label_column (str): Column containing actual values (ground truths).

        date_column (str): Column containing dates for over time metrics.

        decimals (int): Number of decimals to round metrics to. Default is 4.

    Raises:
        AssertionError: If `dataframe` is not a pandas DataFrame, or if `score_column` or
            `label_column` are not strings, or if `score_column` or `label_column` are
            not in dataframe.columns.

    Examples:
        Get binary classification metrics on a labeled and scored dataset:

        >>> import pandas
        >>> from pprint import pprint
        >>> from modelop.monitors.performance import ModelEvaluator

        >>> dataframe = pandas.DataFrame(
        ...     [
        ...         {'id':0, 'score':0, 'label':0}, {'id':1, 'score':1, 'label':1},
        ...         {'id':2, 'score':0, 'label':1}, {'id':3, 'score':1, 'label':0},
        ...         {'id':4, 'score':0, 'label':0}, {'id':5, 'score':1, 'label':0},
        ...         {'id':8, 'score':0, 'label':0}, {'id':7, 'score':1, 'label':1}
        ...     ]
        ... )

        >>> model_evaluator = ModelEvaluator(
        ...     dataframe=dataframe,
        ...     score_column='score',
        ...     label_column='label'
        ... )

        >>> pprint(
        ...     model_evaluator.evaluate_performance(
        ...         pre_defined_metrics='classification_metrics',
        ...         flatten=False
        ...     ),
        ...     sort_dicts=False
        ... )
        {'performance': [{'test_name': 'Classification Metrics',
                          'test_category': 'performance',
                          'test_type': 'classification_metrics',
                          'test_id': 'performance_classification_metrics',
                          'values': {'accuracy': 0.625,
                                     'precision': 0.5,
                                     'recall': 0.6667,
                                     'f1_score': 0.5714,
                                     'auc': 0.6333,
                                     'confusion_matrix': [{'0': 0.375, '1': 0.25},
                                                          {'0': 0.125, '1': 0.25}]}}]}

        Get Multiclass classification metrics on a labeled and scored dataset:

        >>> import pandas
        >>> from pprint import pprint
        >>> from modelop.monitors.performance import ModelEvaluator

        >>> dataframe = pandas.DataFrame(
        ...     [
        ...         {'id':0, 'score':0, 'label':0}, {'id':1, 'score':1, 'label':1},
        ...         {'id':2, 'score':0, 'label':1}, {'id':3, 'score':1, 'label':2},
        ...         {'id':4, 'score':0, 'label':2}, {'id':5, 'score':2, 'label':0},
        ...         {'id':8, 'score':1, 'label':0}, {'id':7, 'score':2, 'label':1}
        ...     ]
        ... )

        >>> model_evaluator = ModelEvaluator(
        ...     dataframe=dataframe,
        ...     score_column='score',
        ...     label_column='label'
        ... )

        >>> pprint(
        ...     model_evaluator.evaluate_performance(
        ...         pre_defined_metrics='classification_metrics',
        ...         flatten=False
        ...     ),
        ...     sort_dicts=False
        ... )
        {'performance': [{'test_name': 'Classification Metrics',
                          'test_category': 'performance',
                          'test_type': 'classification_metrics',
                          'test_id': 'performance_classification_metrics',
                          'values': {'accuracy': 0.25,
                                     'precision': 0.25,
                                     'recall': 0.25,
                                     'f1_score': 0.25,
                                     'confusion_matrix': [{'0': 0.125,
                                                           '1': 0.125,
                                                           '2': 0.125},
                                                          {'0': 0.125,
                                                           '1': 0.125,
                                                           '2': 0.125},
                                                          {'0': 0.125,
                                                           '1': 0.125,
                                                           '2': 0.0}]}}]}

        Get Regression metrics on a labeled and scored dataset:

        >>> dataframe = pandas.DataFrame(
        ...     [
        ...         {'id':0, 'pred':0.2, 'truth':0.3}, {'id':1, 'pred':1.1, 'truth':1.4},
        ...         {'id':2, 'pred':0.5, 'truth':1.0}, {'id':3, 'pred':0.7, 'truth':0.7},
        ...         {'id':4, 'pred':0.6, 'truth':0.5}, {'id':5, 'pred':0.8, 'truth':0.6},
        ...         {'id':8, 'pred':1.2, 'truth':0.9}, {'id':7, 'pred':1.2, 'truth':1.0}
        ...     ]
        ... )

        >>> model_evaluator = ModelEvaluator(
        ...     dataframe=dataframe,
        ...     score_column='pred',
        ...     label_column='truth'
        ... )

        >>> pprint(
        ...     model_evaluator.evaluate_performance(
        ...         pre_defined_metrics='regression_metrics',
        ...         flatten=True
        ...     ),
        ...     sort_dicts=False
        ... )
        {'rmse': 0.2574,
         'mae': 0.2125,
         'r2_score': 0.369,
         'performance': [{'test_name': 'Regression Metrics',
                          'test_category': 'performance',
                          'test_type': 'regression_metrics',
                          'test_id': 'performance_regression_metrics',
                          'values': {'rmse': 0.2574,
                                     'mae': 0.2125,
                                     'r2_score': 0.369}}]}
    """

    def __init__(
        self,
        dataframe: pandas.DataFrame,
        job_json: dict = None,
        label_column: str = None,
        score_column: str = None,
        date_column: str = None,
        decimals: Optional[int] = 4,
        positive_label: Union[float, bool, str] = 1,
    ) -> None:

        input_schema_definition = None
        if job_json is not None:
            logger.info(
                "Parameter 'job_json' is present and will be used to extract "
                "'label_column' and 'score_column'."
            )

            input_schema_definition = infer.extract_input_schema(job_json)
            monitoring_parameters = infer.set_monitoring_parameters(
                schema_json=input_schema_definition, check_schema=True
            )

            label_column = monitoring_parameters["label_column"]
            score_column = monitoring_parameters["score_column"]
            date_column = monitoring_parameters["date_column"]
            if (monitoring_parameters["positive_label"] is not None
                    and type(monitoring_parameters["positive_label"]) is list
                    and len(monitoring_parameters["positive_label"]) > 0):
                positive_label = monitoring_parameters["positive_label"][0]
        else:
            logger.info(
                "Parameter 'job_json' it not present, attempting to use "
                "'label_column' and 'score_column' instead."
            )
            if label_column is None or score_column is None:
                missing_args_error = (
                    "Parameter 'job_json' is not present,"
                    " but one of 'label_column' and 'score_column' was not provided. "
                    "Both 'label_column' and 'score_column' input parameters are"
                    " required if 'job_json' is not provided."
                )
                logger.error(missing_args_error)
                raise Exception(missing_args_error)

        check_input_types(
            inputs=[
                {"label_column": label_column},
                {"score_column": score_column},
            ],
            types=(str),
        )

        if date_column is not None:
            check_input_types(inputs=[{"date_column": date_column}], types=(str))

        assert isinstance(
            dataframe, pandas.DataFrame
        ), "dataframe should be of type (pandas.DataFrame)."

        check_columns_in_dataframe(
            dataframe=dataframe, columns=[score_column, label_column]
        )

        if date_column is not None:
            check_columns_in_dataframe(
                dataframe=dataframe, columns=[score_column, label_column]
            )
            logger.info("Restricting dataframe to columns %s", [score_column, label_column, date_column])
            self.dataframe = dataframe[[score_column, label_column, date_column]]
        else:
            logger.info("Restricting dataframe to columns %s", [score_column, label_column])
            self.dataframe = dataframe[[score_column, label_column]]

        logger.info("Dropping any records with null values in label or score column")
        self.dataframe = check_and_drop_nulls(dataframe, [score_column, label_column])

        if job_json is not None and input_schema_definition is not None:
            logger.info("Verifying score and label data cast")
            self.dataframe = type_check_and_cast(self.dataframe, input_schema_definition)

        self.score_column = score_column
        self.label_column = label_column
        self.date_column = date_column
        self.decimals = decimals
        self.positive_label = positive_label
        self.num_unique_labels = None

    def __str__(self):
        return self.__class__.__name__

    def evaluate_performance(
        self,
        pre_defined_metrics: str = None,
        result_wrapper_key: str = "performance",
        flatten: bool = True,
        include_over_time: bool = True
    ):
        """
        Evaluates the performance of a model on a scored and labeled dataset.

        Args:
            pre_defined_metrics (str): 'regression_metrics' or 'classification_metrics'.

            result_wrapper_key (str): Provides a key to wrap the result dict (flattened results are outside this entry).

            flatten (bool): Surfaces flat metrics to top level of return dictionary.

            include_over_time (bool): Includes a new section with metrics calculation over a prediction date column

        Returns:
            Machine Learning metrics computed on a scored and labeled dataset.
        """

        # Remove capitalization
        pre_defined_metrics = pre_defined_metrics.lower()

        # Make sure choice is valid
        check_pre_defined_metric(
            pre_defined_test=pre_defined_metrics,
            metrics_function="evaluate_performance",
        )

        values_over_time = {}
        y_pred = self.dataframe[self.score_column]
        y_label = self.dataframe[self.label_column]

        if pre_defined_metrics == "regression_metrics":
            test_name = "Regression Metrics"
            values = evaluate_regression(
                y_pred=y_pred,
                y_label=y_label,
                decimals=self.decimals,
            )
            if include_over_time and self.date_column is not None:
                values_over_time = self.performance_over_time(evaluate_regression, ['confusion_matrix'], result_wrapper_key, test_name)

        elif pre_defined_metrics == "classification_metrics":
            test_name = "Classification Metrics"

            # Get number of unique labels from the union of y_true, y_pred.
            self.num_unique_labels = len(unique_labels(y_pred, y_label))

            assert self.num_unique_labels > 0, "No labels were detected!"

            if self.num_unique_labels <= 2:
                values = evaluate_binary_classification(
                    y_pred=y_pred,
                    y_label=y_label,
                    decimals=self.decimals,
                    pos_label=self.positive_label
                )
                if include_over_time and self.date_column is not None:
                    values_over_time = self.performance_over_time(evaluate_binary_classification, ['confusion_matrix'], result_wrapper_key, test_name, self.positive_label)
            else:
                values = evaluate_multiclass_classification(
                    y_pred=y_pred,
                    y_label=y_label,
                    decimals=self.decimals,
                )
                if include_over_time and self.date_column is not None:
                    values_over_time = self.performance_over_time(evaluate_multiclass_classification, ['confusion_matrix'], result_wrapper_key, test_name)

        metrics = {
            "test_name": test_name,
            "test_category": "performance",
            "test_type": pre_defined_metrics.lower().replace("-", "_"),
            "test_id": "performance_{}".format(pre_defined_metrics),
            "values": values,
        }

        result = {}
        if flatten:
            if (pre_defined_metrics == "classification_metrics") and (
                self.num_unique_labels <= 2
            ):
                result.update(
                    {
                        # Top-level metrics
                        "accuracy": metrics["values"]["accuracy"],
                        "precision": metrics["values"]["precision"],
                        "recall": metrics["values"]["recall"],
                        "auc": metrics["values"]["auc"],
                        "f1_score": metrics["values"]["f1_score"],
                        "confusion_matrix": metrics["values"]["confusion_matrix"],
                    }
                )
            elif (pre_defined_metrics == "classification_metrics") and (
                self.num_unique_labels > 2
            ):
                result.update(
                    {
                        # Top-level metrics
                        "accuracy": metrics["values"]["accuracy"],
                        "precision": metrics["values"]["precision"],
                        "recall": metrics["values"]["recall"],
                        "f1_score": metrics["values"]["f1_score"],
                        "confusion_matrix": metrics["values"]["confusion_matrix"],
                    }
                )
            elif pre_defined_metrics == "regression_metrics":
                result.update(
                    {
                        # Top-level metrics
                        "rmse": metrics["values"]["rmse"],
                        "mae": metrics["values"]["mae"],
                        "r2_score": metrics["values"]["r2_score"],
                    }
                )
        if include_over_time and self.date_column is not None:
            result.update(values_over_time)
        # Vanilla ModelEvaluator output
        result[result_wrapper_key] = [metrics]

        return result

    def performance_over_time(self, evaluation_function,
                              exclude_metrics: list = (),
                              result_wrapper_key: str = 'performance',
                              test_name: str = '',
                              pos_label: Union[float, bool, str] = 1
                              ) -> dict:
        """
        Computes the performance metrics as given by the evaluation_function, attempting to split the data by date
        :param evaluation_function: The function to run per data split (by date)
        :param exclude_metrics: A list of keys to exclude from the result of the evaluation_function
        :param result_wrapper_key: The key used for the wrapping the whole test results. This will create a similar key
        :param test_name: The name of the specific test actually run.
        :param pos_label: The value to use as positive class label (for binary classification)
        :return: A dictionary with a graph structure over time
        """
        if self.date_column is not None and self.date_column in self.dataframe:
            self.dataframe = self.dataframe.set_index(check_date_column(self.dataframe, self.date_column).dt.date)
            self.dataframe[self.date_column] = check_date_column(self.dataframe, self.date_column).dt.date
            self.dataframe = self.dataframe.sort_index()

            unique_dates = self.dataframe[self.date_column].unique()

            data = {}
            for date in unique_dates:
                data_of_the_day = self.dataframe.loc[[date]]
                dated_y_pred = data_of_the_day[self.score_column]
                dated_y_label = data_of_the_day[self.label_column]
                if not isinstance(dated_y_pred, pandas.Series):
                    dated_y_pred = pandas.Series(dated_y_pred)
                if not isinstance(dated_y_label, pandas.Series):
                    dated_y_label = pandas.Series(dated_y_label)
                if pos_label != 1:
                    dated_values = evaluation_function(dated_y_pred, dated_y_label, self.decimals, pos_label=pos_label)
                else:
                    dated_values = evaluation_function(dated_y_pred, dated_y_label, self.decimals)

                str_date = str(date)
                for metric in dated_values:
                    if metric in exclude_metrics:
                        continue
                    if metric not in data:
                        data[metric] = []
                    data[metric].append([str_date, dated_values[metric]])

            over_time_key = result_wrapper_key + "_over_time"
            return {over_time_key: {
                "title": "Performance Over Time" + (" - " + test_name if test_name else ""),
                "x_axis_label": "Day",
                "y_axis_label": "Metric",
                "data": data
            },
                "firstPredictionDate": str(unique_dates.min()),
                "lastPredictionDate": str(unique_dates.max()),
            }
        else:
            return {}


def evaluate_binary_classification(
    y_pred: pandas.Series, y_label: pandas.Series, decimals: Optional[int] = 4, pos_label: Union[float, bool, str] = 1
) -> dict:
    """
    Computes accuracy, precision, recall, f1_score, AUC, and confusion matrix.

    Args:
        y_pred (pandas.Series): predictions(scores).

        y_label (pandas.Series): ground_truths(labels).

        decimals (int): Number of decimals to round metrics to. Default is 4.

        pos_label (int, bool, str): Value used as positive label class

    Raises:
        ValueError: If AUC cannot be computed.

    Returns:
        A dictionary of classification metrics.
    """

    # Calculate classification metrics on sample and baseline DFs
    precision = precision_score(y_label, y_pred, pos_label=pos_label)
    recall = recall_score(y_label, y_pred, pos_label=pos_label)
    accuracy = accuracy_score(y_label, y_pred)
    f_1 = f1_score(y_label, y_pred, pos_label=pos_label)
    try:
        if not is_numeric_dtype(y_label):
            y_label_numeric = y_label.map(lambda x: int(x == pos_label), na_action='ignore')
            y_pred_numeric = y_pred.map(lambda x: int(x == pos_label), na_action='ignore')
            auc = numpy.round(roc_auc_score(y_label_numeric, y_pred_numeric), decimals)
        else:
            auc = numpy.round(roc_auc_score(y_label, y_pred), decimals)
    except ValueError:
        auc = None

    # Confusion Matrix
    labels_sorted = sorted(pandas.Series(y_label).unique())
    conf_mat = confusion_matrix(
        y_true=y_label, y_pred=y_pred, normalize="all", labels=labels_sorted
    ).round(decimals)

    label_strings = [str(label) for label in labels_sorted]

    # conf_mat is a numpy array. We turn it into array of dicts
    conf_mat_json = []
    for idx, _ in enumerate(labels_sorted):
        conf_mat_json.append(dict(zip(label_strings, conf_mat[idx, :].tolist())))

    metrics = {
        "accuracy": numpy.round(accuracy, decimals),
        "precision": numpy.round(precision, decimals),
        "recall": numpy.round(recall, decimals),
        "f1_score": numpy.round(f_1, decimals),
        "auc": auc,
        "confusion_matrix": conf_mat_json,
    }

    return metrics


def evaluate_multiclass_classification(
    y_pred: pandas.Series,
    y_label: pandas.Series,
    average: Optional[str] = "weighted",
    decimals: Optional[int] = 4,
) -> dict:
    """
    Computes accuracy, precision, recall, f1_score, AUC, and confusion matrix.

    Args:
        y_pred (pandas.Series): predictions(scores).

        y_label (pandas.Series): ground_truths(labels).

        average (str): averaging method for computing precision, recall, f_1 score. Default is "weighted"

        decimals (int): Number of decimals to round metrics to. Default is 4.

    Raises:
        ValueError: If AUC cannot be computed.

    Returns:
        A dictionary of classification metrics.
    """

    labels_sorted = unique_labels(y_label, y_pred)

    # Calculate classification metrics on sample and baseline DFs
    precision = numpy.round(precision_score(
        y_true=y_label, y_pred=y_pred, average=average, labels=labels_sorted
    ), decimals)
    recall = numpy.round(recall_score(
        y_true=y_label, y_pred=y_pred, average=average, labels=labels_sorted
    ), decimals)
    accuracy = numpy.round(accuracy_score(y_true=y_label, y_pred=y_pred), decimals)
    f_1 = numpy.round(f1_score(
        y_true=y_label, y_pred=y_pred, average=average, labels=labels_sorted
    ), decimals)

    # Confusion Matrix
    conf_mat = numpy.round(confusion_matrix(
        y_true=y_label, y_pred=y_pred, normalize="all", labels=labels_sorted
    ), decimals)

    label_strings = [str(label) for label in labels_sorted]

    # conf_mat is a numpy array. We turn it into array of dicts
    conf_mat_json = []
    for idx, _ in enumerate(labels_sorted):
        conf_mat_json.append(dict(zip(label_strings, conf_mat[idx, :].tolist())))

    if isinstance(precision, numpy.ndarray):
        precision = precision.tolist()
    if isinstance(recall, numpy.ndarray):
        recall = recall.tolist()
    if isinstance(f_1, numpy.ndarray):
        f_1 = f_1.tolist()

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f_1,
        "confusion_matrix": conf_mat_json,
    }

    return metrics


def evaluate_regression(
    y_pred: pandas.Series,
    y_label: pandas.Series,
    decimals: Optional[int] = 4,
) -> dict:
    """
    Computes RMSE, MAE, and r2_score.

    Args:
        y_pred (pandas.Series): predictions(scores).

        y_label (pandas.Series): ground_truths(labels).

        decimals (int): Number of decimals to round metrics to. Default is 4.

    Returns:
        A dictionary of regression metrics.
    """

    # Calculate regression metrics on sample and baseline DFs
    rmse = numpy.sqrt(numpy.mean((y_label - y_pred) ** 2))
    mae = mean_absolute_error(y_label, y_pred)
    _r2_score = r2_score(y_label, y_pred)

    metrics = {
        "rmse": numpy.round(rmse, decimals),
        "mae": numpy.round(mae, decimals),
        "r2_score": numpy.round(_r2_score, decimals) if not math.isnan(_r2_score) else None,
    }

    return metrics


if __name__ == "__main__":
    print(doctest.testmod())
    print()
