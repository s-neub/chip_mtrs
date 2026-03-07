"""A collection of assertion functions to check input types, memberships of columns to \
    DataFrames, etc.

.. toctree::

.. autosummary::
   :toctree: _autosummary
"""
import doctest
from typing import List, Optional

import logging
import pandas

logger = logging.getLogger(__name__)

def check_baseline_and_sample(
    df_baseline: pandas.DataFrame,
    df_sample: pandas.DataFrame,
    check_column_equality: Optional[bool] = True,
):
    """
    A function to check that two dataframes match on column names and types.

    Args:
        df_baseline (pandas.DataFrame): Baseline DataFrame.

        df_sample (pandas.DataFrame): Sample DataFrame.

    Raises:
        AssertionError
    """

    assert isinstance(
        df_baseline, pandas.DataFrame
    ), "df_baseline should be of type <pandas.DataFrame>."

    assert isinstance(
        df_sample, pandas.DataFrame
    ), "df_sample should be of type <pandas.DataFrame>."

    if check_column_equality:

        assert len(df_baseline.columns) == len(
            df_sample.columns
        ), "df_baseline and df_sample should have the same number of columns."

        assert set(df_baseline.columns) == set(
            df_sample.columns
        ), "df_baseline and df_sample should have the same column names."

        assert all(
            df_baseline.dtypes == df_sample.dtypes
        ), "df_baseline and df_sample should have the same column types."


def check_columns_in_dataframe(dataframe: pandas.DataFrame, columns: List[str]):
    """
    A function to check that columns are in DataFrame.

    Args:
        dataframe (pandas.DataFrame): Input DataFrame.

        columns List[str]: List of columns to check.

    Raises:
        KeyError if a column in `columns` is not in `dataframe.columns`
    """

    for column in columns:
        if not column in dataframe.columns:
            raise KeyError("{} does not exist in dataframe.".format(column))


def check_pre_defined_metric(pre_defined_test, metrics_function):
    """
    A function to check that pre_defined_test choice is valid.

    Args:
        pre_defined_test (str): choice of pre_defined_test.

        metrics_function (str): metrics_function under consideration.

    Raises:
        AssertionError
    """

    builtin_metrics = {
        "calculate_drift": [
            "jensen-shannon",
            "js",
            "kolmogorov-smirnov",
            "ks",
            "epps-singleton",
            "es",
            "kullback-leibler",
            "kl",
            "wasserstein",
            "ws",
            "describe",
            "summary",
        ],
        "calculate_concept_drift": [
            "jensen-shannon",
            "js",
            "kolmogorov-smirnov",
            "ks",
            "epps-singleton",
            "es",
            "kullback-leibler",
            "kl",
            "wasserstein",
            "ws",
            "describe",
            "summary",
        ],
        "evaluate_performance": ["regression_metrics", "classification_metrics"],
        "compute_group_metrics": ["aequitas_group"],
        "compute_bias_metrics": ["aequitas_bias"],
    }

    assert (
        pre_defined_test in builtin_metrics[metrics_function]
    ), "pre_defined_test should be one of {}".format(
        str(builtin_metrics[metrics_function])
    )


def check_input_types(inputs: List[dict], types: tuple):
    """
    A function to check that inputs are the correct type(s).

    Args:
        inputs (List[dict]): Inputs to check.

        types (tuple): Acceptable type(s).

    Raises:
        AssertionError

    Example:
        The following checks if score_column and label_column are strings

        >>> score_column = "prediction"
        >>> label_column = "ground_truth"

        >>> check_input_types(
        ...     inputs=[
        ...         {"score_column": score_column},
        ...         {"label_column": label_column}
        ...     ],
        ...     types=(str),
        ... )
    """

    for input_dict in inputs:
        var_name = next(iter(input_dict.keys()))
        var_value = next(iter(input_dict.values()))
        assert isinstance(var_value, types), "{} should be of type {}.".format(
            var_name, types
        )

def check_date_column(dataframe, date_column):
    """
    A function to check that date_column only contains valid datetime formats.

    Args:
        dataframe (pandas.Dataframe): Dataframe to check for valid date_column values.

        date_column (str): name/label of the date_column to check.

    Returns:
        dataframe (pandas.Series): returns the series of the date column converted to pandas datetime.

    Raises:
        Exception if the date column couldn't be parsed as date.
    """
    try:
        return pandas.to_datetime(dataframe[date_column])
    except Exception as err:
        err_message = str(err.args)
        logger.error(f"Failed to convert date_column to standard pandas datetimes: {err_message}")
        raise ValueError(f"Failed to convert date_column to standard pandas datetimes:\n\t{err_message}") from err


if __name__ == "__main__":
    print(doctest.testmod())
    print()
