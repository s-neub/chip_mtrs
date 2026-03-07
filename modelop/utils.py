"""
Tools for creating production-ready models.
"""
import doctest
import logging
import operator
from typing import List

import numpy
import pandas


def configure_logger(
    _format: str = "[%(asctime)s] %(levelname)s - %(message)s",
    level: str = "INFO",
    datefmt="%Y-%m-%d %H:%M:%S %z",
) -> logging.Logger:
    """
    Wrapper for `logging.basicConfig` to configure a Python logger. The defaults are recommended
    for logging and printing messages during a production model job or REST deployment.

    Args:
        _format (str, optional): The logging format.
            Defaults to "[%(asctime)s] %(levelname)s - % (message)s".
        level (str, optional): The log level to print.
            Defaults to "INFO".
        datefmt (str, optional): The datetime format for the log messages.
            Defaults to "%Y-%m-%d %H:%M:%S %z".

    Example:
        >>> import modelop.utils
        >>> logger = modelop.utils.configure_logger()
        >>> logger.warning("This is a warning!")
        [2021-05-10 09:49:07 -0500] WARNING - This is a warning!
        >>> logger.info("A general info message")
        [2021-05-10 09:51:11 -0500] INFO - A general info message
    """
    if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError("Unknown log level: {}".format(level))

    # Map string log level to logging enum
    if level == "DEBUG":
        log_level = logging.DEBUG
    elif level == "INFO":
        log_level = logging.INFO
    elif level == "WARNING":
        log_level = logging.WARNING
    elif level == "ERROR":
        log_level = logging.ERROR
    elif level == "CRITICAL":
        log_level = logging.CRITICAL

    logging.basicConfig(
        format=_format,
        level=log_level,
        datefmt=datefmt,
    )

    return logging.getLogger()


def merge(*args):
    """
    Special merge function that appends to existing arrays inside the dictionary rather than overwriting them.
    As opposed to dict `update` method, this `merge` appends item value into the collisioned key

    Args:
        args: Arguments are dict objects to be merged together

    Example:

        >>> from pprint import pprint

        >>> merge({"collision_key": [{"elem1": "value1"}]}, {"collision_key": [{"elem2": "value2"}]})
        {'collision_key': [{'elem1': 'value1'}, {'elem2': 'value2'}]}

        >>> pprint(
        ...     merge(
        ...         {"data_drift": [{"test_name": "Jensen-Shannon","test_category": "data_drift","test_type": "jensen_shannon","metric": "distance","values": {"ownership": 0.17, "amount": 0.1362}}]},
        ...         {"data_drift": [{"test_name": "Kolmogorov-Smirnov","test_category": "data_drift","test_type": "kolmogorov_smirnov","metric": "p_value","values": {"amount": 0.2701, "num_cards": 0.9888}}]},
        ...     ),
        ...     sort_dicts=False
        ... )
        {'data_drift': [{'test_name': 'Jensen-Shannon',
                         'test_category': 'data_drift',
                         'test_type': 'jensen_shannon',
                         'metric': 'distance',
                         'values': {'ownership': 0.17, 'amount': 0.1362}},
                        {'test_name': 'Kolmogorov-Smirnov',
                         'test_category': 'data_drift',
                         'test_type': 'kolmogorov_smirnov',
                         'metric': 'p_value',
                         'values': {'amount': 0.2701, 'num_cards': 0.9888}}]}
    """
    wrapper = {}
    for metric in args:
        for key in metric.keys():
            over_time_count_on_merge = 1
            if key in wrapper.keys():
                x = wrapper[key]
                y = metric[key]
                if type(x) is list:
                    if type(y) is list:
                        wrapper[key] = x + y
                    else:
                        wrapper[key].append(y)
                else:
                    if type(y) is list:
                        wrapper[key] = [x] + y
                    elif key in ["firstPredictionDate", "lastPredictionDate"]:
                        pass
                    elif "over_time" in key:
                        new_key = f"{key}_{over_time_count_on_merge}"
                        over_time_count_on_merge += 1
                        wrapper[new_key] = y
                    else:
                        wrapper[key] = [x, y]
            else:
                wrapper[key] = metric[key]
    wrapper = dict(sorted(wrapper.items(), key=lambda x: 0 if "over_time" in x[0] else 1))
    return wrapper


def fix_numpy_nans_in_dict(dictionary: dict) -> dict:
    """A function to iterate over values in a dictionary,
    and change all numpy.nan values to a python None.

    Args:
        values (dict): Input dict to fix.

    Returns:
       dict: Fixed dict.
    """

    # This will hold return dict
    fixed_values = {}
    for key, val in dictionary.items():
        # Some values are strings, skip over them (no need to fix)
        try:
            # If value is numeric, check for numpy.nan;
            # If True, change to None, else keep unchanged
            if numpy.isnan(val):
                val = None
            fixed_values[key] = val
        except TypeError:
            fixed_values[key] = val

    return fixed_values

def check_and_drop_nulls(dataframe: pandas.DataFrame, columns: List[str]):
    """
    A function to check for and drop NULLs in given columns.

    Args:
        dataframe (pandas.DataFrame): Pandas DataFrame of data, most likely scored and labeled.

        columns (List[str]): List of column names to check for null value.

    Returns:
        pandas.DataFrame: Copy of the dataframe with dropped nulls.
    """
    # Checking for NULLs in columns
    for column in columns:
        if column in dataframe.columns:
            null_count = dataframe[column].isna().sum()
            if null_count > 0:
                dataframe = dataframe.dropna(subset=[column])

    return dataframe


def fix_numpy_nans_in_dict_array(dict_array: List[dict]) -> List[dict]:
    """A function to iterate over dictionaries in an array,
    and change all numpy.nan values to a python None.

    Args:
        values (List[dict]): Input list of dicts to fix.

    Returns:
        List[dict]: Fixed list.
    """

    # This will hold return list
    fixed_values = []
    # Iterate over dicts in list
    for dictionary in dict_array:
        fixed_dictionary = {}
        # Iterate over items in dict
        for key, val in dictionary.items():
            # Some values are strings, skip over them (no need to fix)
            try:
                # If value is numeric, check for numpy.nan;
                # If True, change to None, else keep unchanged
                if numpy.isnan(val):
                    val = None
                fixed_dictionary[key] = val
            except TypeError:
                fixed_dictionary[key] = val
        # Dictionary is now fixed. Add to return list
        fixed_values.append(fixed_dictionary)

    return fixed_values


def get_min_max_values_keys_from_dict(values_dict: dict) -> dict:
    """A function to iterate over a dictionary of numerical values (with possible None(s)),
    and return min/max key/values.

    Args:
        values_dict (dict): Input dictionary.

    Returns:
        dict: A dict containing max_feature, max_value, min_feature, min_value.

    Examples:
        Get min/max /key/value tuples from dict of values:

        >>> d = {'a':1, 'b':2, 'c':None}
        >>> get_min_max_values_features_from_dict(values_dict=d)
        {'min_feature': 'a', 'min_value': 1, 'max_feature': 'b', 'max_value': 2}

    """

    assert values_dict != {}, "values_dict must not be empty!"

    values_dict_no_nulls = {k: v for k, v in values_dict.items() if v is not None}

    if values_dict_no_nulls == {}:
        (max_feature, max_value) = (list(values_dict.keys())[0], None)
        (min_feature, min_value) = (max_feature, max_value)

    else:
        (max_feature, max_value) = max(
            values_dict_no_nulls.items(), key=operator.itemgetter(1)
        )
        (min_feature, min_value) = min(
            values_dict_no_nulls.items(), key=operator.itemgetter(1)
        )

    return {
        "min_feature": min_feature,
        "min_value": min_value,
        "max_feature": max_feature,
        "max_value": max_value,
    }

def clean_scoring_optional_fields(dataframe: pandas.DataFrame,
                                  input_schema_definition: dict,
                                  numerical_columns: list = [],
                                  categorical_columns: list = []
                                  ) -> tuple:
    """A function to iterate over values in a job schema from infer schema,
    and remove them from numerical and or categorical columns.

    Args:
        dataframe (Dataframe): Input dataframe to to check against for columns.
        input_schema_definition (Dict): Input schema definition from infer functions.
        numerical_columns (List[str]): The current list of numerical columns.
        categorical_columns (List[str]): The current list of categorical columns.


    Returns:
       List[str]: list of numerical columns.
       List[str]: list of categorical columns.
    """

    for field in input_schema_definition['fields']:
        if "scoringOptional" in field and field["scoringOptional"] and field["name"] not in dataframe.columns:
            if field["name"] in numerical_columns:
                numerical_columns.remove(field["name"])
            elif field["name"] in categorical_columns:
                categorical_columns.remove(field["name"])
    return numerical_columns, categorical_columns

if __name__ == "__main__":
    print(doctest.testmod())
    print()
