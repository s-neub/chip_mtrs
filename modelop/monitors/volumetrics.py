"""
This module provides several volumetric monitors for governing input and output \
data slices of machine learning models.

See `VolumetricMonitor` for usage examples.

.. toctree::

.. autosummary::
   :toctree: _autosummary
"""
import collections
import doctest
import logging
from typing import List, Optional, Union

# Third party packages
import pandas

from modelop.utils import clean_scoring_optional_fields
import modelop.schema.infer as infer
from modelop.monitors import assertions
from modelop.monitors.drift import fix_numpy_nans_and_infs_in_dict, summarize_df

pandas.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


class VolumetricMonitor:
    """
    Monitors the row counts of one or more datasets.

    Each volumetric summary method (`count`, `count_compared_to`, etc.) is given as a
    class instance method. Because each summary uses at least one DataFrame, the first is passed
    to the monitor constructor. If more DataFrames are needed for comparison, they are given to the
    instance methods.

    Args:
        dataframe (pandas.DataFrame): The first (or only) DataFrame for volumetric monitoring

        decimals (int): Number of decimals to round metrics to. Default is 4.

    Raises:
        TypeError: If `dataframe` is not a pandas DataFrame or if `schema` is not `None`
            and not a `dict`

    Examples:
        Get a basic count dict for a data set:

        >>> import pandas
        >>> from modelop.monitors.volumetrics import VolumetricMonitor
        >>> from pprint import pprint

        >>> dataframe = pandas.DataFrame({"values": [1, 2, 3]})
        >>> pprint(
        ...     VolumetricMonitor(dataframe).count(flatten=False),
        ...     sort_dicts=False
        ... )
        {'volumetrics': [{'test_name': 'Count',
                          'test_category': 'volumetrics',
                          'test_type': 'count',
                          'test_id': 'volumetrics_count',
                          'values': {'record_count': 3}}]}

        Compare the row counts of two datasets:

        >>> df1 = pandas.DataFrame({"values": [1, 2, 3, 4, 5]})
        >>> df2 = pandas.DataFrame({"values": [1, 2, 3]})
        >>> comparison = VolumetricMonitor(df1).count_comparison(
        ...     df2,
        ...     flatten=False)
        >>> pprint(
        ...     comparison,
        ...     sort_dicts=False
        ... )
        {'volumetrics': [{'test_name': 'Count Comparison',
                          'test_category': 'volumetrics',
                          'test_type': 'count_comparison',
                          'test_id': 'volumetrics_count_comparison',
                          'values': {'dataframe_1_record_count': 5,
                                     'dataframe_2_record_count': 3,
                                     'record_count_difference': 2}}]}
    """

    def __init__(
        self, dataframe: pandas.DataFrame, decimals: Optional[int] = 4
    ) -> None:
        if not isinstance(dataframe, pandas.DataFrame):
            raise TypeError("dataframe must be a pandas.DataFrame instance")

        #: pandas.DataFrame: The first (or only) DataFrame for volumetric monitoring
        self.dataframe = dataframe
        self.decimals = decimals

    def summary(
        self,
        job_json: dict = None,
        categorical_columns: List[str] = None,
        numerical_columns: List[str] = None,
        result_wrapper_key: str = "volumetrics",
        date_column: str = None,
        include_over_time: bool = True
    ) -> dict:
        """
        Describes numerical and categorical columns in `self.dataframe`.

        Args:
            job_json (dict): JSON dictionary with the metadata of the model.

            categorical_columns (List[str]): Categorical column names.

            numerical_columns (List[str]): Numerical column names.
            
            date_column (str): Column containing dates for over time metrics.

            result_wrapper_key (str): Provides a key to wrap the result dict (flattened results are outside this entry).

            include_over_time (bool): Includes a new section with metrics calculation over a prediction date column

        Returns:
            dict: The summary test result. ::

                {
                    "test_name": "Summary",
                    'test_category': 'volumetrics',
                    "test_type": "summary",
                    'test_id': 'volumetrics_summary',
                    "values": {
                        "column_1": {}, # Pandas summary output
                        # ...
                    },
                }
        """

        if job_json is not None:
            logger.info(
                "Parameter 'job_json' is present and will be used to extract "
                "'categorical_columns' and 'numerical_columns'."
            )
            input_schema_definition = infer.extract_input_schema(job_json)
            monitoring_parameters = infer.set_monitoring_parameters(
                schema_json=input_schema_definition, check_schema=True
            )
            categorical_columns = monitoring_parameters["categorical_columns"]
            numerical_columns = monitoring_parameters["numerical_columns"]
            date_column = monitoring_parameters["date_column"]
            # add score and/or label columns, if present
            score_column = monitoring_parameters["score_column"]
            if score_column:
                if (
                    monitoring_parameters["feature_dataclass"][score_column]
                    == "categorical"
                ):
                    categorical_columns.append(score_column)
                else:
                    numerical_columns.append(score_column)

            label_column = monitoring_parameters["label_column"]
            if label_column:
                if (
                    monitoring_parameters["feature_dataclass"][label_column]
                    == "categorical"
                ):
                    categorical_columns.append(label_column)
                else:
                    numerical_columns.append(label_column)
            numerical_columns, categorical_columns = clean_scoring_optional_fields(
                self.dataframe,
                input_schema_definition,
                numerical_columns,
                categorical_columns,
            )
        else:
            logger.info(
                "Parameter 'job_json' it not present, attempting to use "
                "'categorical_columns' and 'numerical_columns' instead."
            )
            if categorical_columns is None or numerical_columns is None:
                missing_args_error = (
                    "Parameter 'job_json' is not present,"
                    " but one of 'categorical_columns' and 'numerical_columns' was not provided. "
                    "Both 'categorical_columns' and 'numerical_columns' input parameters are"
                    " required if 'job_json' is not provided."
                )
                logger.error(missing_args_error)
                raise Exception(missing_args_error)
            
        if date_column is not None:
            assertions.check_input_types(inputs=[{"date_column": date_column}], types=(str))

        if date_column is not None:
            assertions.check_columns_in_dataframe(
                dataframe=self.dataframe, columns=[date_column]
            )

        values_over_time = {}

        values = summarize_df(
            dataframe=self.dataframe,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            decimals=self.decimals,
        )

        values["categorical_summary"] = values["categorical_summary"].to_dict()
        values["numerical_summary"] = values["numerical_summary"].to_dict()

        if include_over_time and date_column is not None:
            if date_column is not None and date_column in self.dataframe:
                dataframe = self.dataframe.set_index(assertions.check_date_column(self.dataframe, date_column).dt.date)
                dataframe[date_column] = assertions.check_date_column(dataframe, date_column).dt.date
                dataframe = dataframe.sort_index()

            unique_dates = dataframe[date_column].unique()

            data_over_time = {}
            for date in unique_dates:
                data_of_the_day = dataframe.loc[[date]]
                dated_values = summarize_df(
                    dataframe=data_of_the_day,
                    numerical_columns=numerical_columns,
                    categorical_columns=categorical_columns,
                    decimals=self.decimals
                )

                str_date = str(date)
                for metric in dated_values:
                    rows = dated_values[metric].index
                    if {"count", "std", "mean"}.issubset(rows):
                        flattened = flatten_json(dated_values[metric].loc[["count", "std", "mean", "count_of_nulls"]].to_dict())
                    elif {"count"}.issubset(rows):
                        flattened = flatten_json(dated_values[metric].loc[["count", "count_of_nulls"]].to_dict())
                    # flattened = flatten_json(dated_values[metric].to_dict())
                    for key in flattened.keys():
                        if key not in data_over_time:
                            data_over_time[key] = []
                        data_over_time[key].append([str_date, flattened[key]])

            over_time_key = result_wrapper_key + "_over_time"
            values_over_time = {
                over_time_key: {
                "title": "Volumetrics Summary Over Time",
                "x_axis_label": "Day",
                "y_axis_label": "Metric",
                "data": data_over_time
            },
                "firstPredictionDate": str(unique_dates.min()),
                "lastPredictionDate": str(unique_dates.max()),
            }

        # Replace numpy.nan to None in metrics dictionaries (if present)
        for feature, feature_dict in values["categorical_summary"].items():
            values["categorical_summary"][feature] = fix_numpy_nans_and_infs_in_dict(
                values=feature_dict, test_name="Summary"
            )

        for feature, feature_dict in values["numerical_summary"].items():
            values["numerical_summary"][feature] = fix_numpy_nans_and_infs_in_dict(
                values=feature_dict, test_name="Summary"
            )

        summary_results = {
            "test_name": "Summary",
            "test_category": "volumetrics",
            "test_type": "summary",
            "test_id": "volumetrics_summary",
            "values": values,
        }
        results = values_over_time if values_over_time else {}
        results.update({result_wrapper_key: [summary_results]})

        return results

    def count(
        self,
        result_wrapper_key: str = "volumetrics",
        flatten: bool = True,
        job_json: dict = None,
        date_column: str = None,
        include_over_time: bool = True
    ) -> dict:
        """
        Counts the number of rows in `self.dataframe`

        Args:
            job_json (dict): JSON dictionary with the metadata of the model.
            
            date_column (str): Column containing dates for over time metrics.

            flatten (bool): Surfaces flat metrics to top level of return dictionary.

            result_wrapper_key (str): Provides a key to wrap the result dict (flattened results are outside this entry).

            include_over_time (bool): Includes a new section with metrics calculation over a prediction date column

        Returns:
            dict: The record count as a test result. ::

                {
                    "test_name": "Count",
                    'test_category': 'volumetrics',
                    "test_type": "count",
                    'test_id': 'volumetrics_count',
                    "values": {
                        "record_count": 10
                    },
                }
        """

        if job_json is not None:
            logger.info(
                "Parameter 'job_json' is present and will be used to extract "
                "'date_column if present'"
            )
            input_schema_definition = infer.extract_input_schema(job_json)
            monitoring_parameters = infer.set_monitoring_parameters(
                schema_json=input_schema_definition, check_schema=True
            )
            date_column = monitoring_parameters["date_column"]
            
        if date_column is not None:
            assertions.check_input_types(inputs=[{"date_column": date_column}], types=(str))

        if date_column is not None:
            assertions.check_columns_in_dataframe(
                dataframe=self.dataframe, columns=[date_column]
            )

        values_over_time = {}

        count_results = {
            "test_name": "Count",
            "test_category": "volumetrics",
            "test_type": "count",
            "test_id": "volumetrics_count",
            "values": {"record_count": len(self.dataframe.index)},
        }

        if include_over_time and date_column is not None:
            if date_column is not None and date_column in self.dataframe:
                dataframe = self.dataframe.set_index(assertions.check_date_column(self.dataframe, date_column).dt.date)
                dataframe[date_column] = assertions.check_date_column(dataframe, date_column).dt.date
                dataframe = dataframe.sort_index()

            unique_dates = dataframe[date_column].unique()

            data_over_time = {}
            data_over_time["record_count"] = []
            for date in unique_dates:
                data_of_the_day = dataframe.loc[[date]]
                dated_values = len(data_of_the_day.index)
                str_date = str(date)
                data_over_time["record_count"].append([str_date, dated_values])

            over_time_key = result_wrapper_key + "_over_time"
            values_over_time = {
                over_time_key: {
                "title": "Volumetrics Count Over Time",
                "x_axis_label": "Day",
                "y_axis_label": "Metric",
                "data": data_over_time
            },
                "firstPredictionDate": str(unique_dates.min()),
                "lastPredictionDate": str(unique_dates.max()),
            }
        result = values_over_time if values_over_time else {}
        if flatten:
            result.update({
                "record_count": count_results["values"]["record_count"],
                "allVolumetricMonitorRecordCount": count_results["values"]["record_count"],
                result_wrapper_key: [count_results],
            })
        else:
            result.update({result_wrapper_key: [count_results]})
        return result

    def count_comparison(
        self,
        dataframe_2: pandas.DataFrame,
        result_wrapper_key: str = "volumetrics",
        flatten=True,
    ) -> dict:
        """
        Compares the record counts of `self.dataframe` to `dataframe_2`.

        Args:
            dataframe_2: The second DataFrame to compare

            result_wrapper_key (str): Provides a key to wrap the result dict (flattened results are outside this entry).

            flatten (bool): Surfaces flat metrics to top level of return dictionary.

        Returns:
            A dictionary with the record counts of `self.dataframe`, `dataframe_2`
            and the difference between `self.dataframe` and `dataframe_2`::

                {
                    "test_name": "Count Comparison",
                    'test_category': 'volumetrics',
                    "test_type": "count_comparison",
                    'test_id': 'volumetrics_count_comparison',
                    "values": {
                        "dataframe_1_record_count": 10,
                        "dataframe_2_record_count": 8,
                        "record_count_difference": 2,
                    }
                }

        Raises:
            TypeError: If `dataframe_2` is not a pandas DataFrame
        """
        if not isinstance(dataframe_2, pandas.DataFrame):
            raise TypeError("dataframe_2 must be a pandas.DataFrame instance")

        dataframe_1_count = len(self.dataframe.index)
        dataframe_2_count = len(dataframe_2.index)
        difference = dataframe_1_count - dataframe_2_count

        count_comparison_results = {
            "test_name": "Count Comparison",
            "test_category": "volumetrics",
            "test_type": "count_comparison",
            "test_id": "volumetrics_count_comparison",
            "values": {
                "dataframe_1_record_count": dataframe_1_count,
                "dataframe_2_record_count": dataframe_2_count,
                "record_count_difference": difference,
            },
        }

        if flatten:
            result = {
                "record_count_difference": count_comparison_results["values"][
                    "record_count_difference"
                ],
                result_wrapper_key: [count_comparison_results],
            }
        else:
            result = {result_wrapper_key: [count_comparison_results]}

        return result

    def identifier_comparison(
        self,
        dataframe_2: pandas.DataFrame,
        job_json: dict = None,
        identifier_columns: Union[str, List[str]] = None,
        result_wrapper_key: str = "volumetrics",
        flatten: bool = True,
    ) -> dict:
        """
        Compares the count of unique identifiers in `identifier_columns`
        between `self.dataframe` and `dataframe_2`.

        Args:
            dataframe_2: The second DataFrame to compare with.

            job_json (dict): JSON dictionary with the metadata of the model.

            identifier_columns: The column name(s) that contains the unique identifiers
            in `self.dataframe` and `dataframe_2`. If a list, the values at each column are
            combined into one identifier.

            result_wrapper_key (str): Provides a key to wrap the result dict (flattened results are outside this entry).

            flatten (bool): Surfaces flat metrics to top level of return dictionary.


        Returns:
            A dictionary with the record counts of both DataFrames and a breakdown of identifier
            counts that are in one DataFrame but not the other. If the identifier counts between
            the DataFrames are identical, the key `identifiers_match` is set to True
            (False otherwise).::

                {
                    "test_name": "Identifier Comparison",
                    'test_category': 'volumetrics',
                    "test_type": "identifier_comparison",
                    'test_id': 'volumetrics_identifier_comparison',
                    "values": {
                        "identifiers_match": True,
                        "dataframe_1": {
                            "identifier_column": "column_name",
                            "record_count": 10,
                            "unique_identifier_count": 10,
                            "extra_identifiers": {
                                "total": 0,
                                "breakdown": {}
                            }
                        },
                        "dataframe_2": {
                            "identifier_column": "column_name",
                            "record_count": 10,
                            "unique_identifier_count": 10,
                            "extra_identifiers": {
                                "total": 0,
                                "breakdown": {}
                            }
                        }
                    }
                }

        Raises:
            TypeError: If `dataframe` is not a pandas DataFrame or identifier_columns is not in
                (str, list, tuple)
            KeyError: If `identifier_columns` is not in `self.dataframe` and `dataframe_2`

        Examples:
            Compare two DataFrames with one-to-one matches between identifiers in the `id` column::

                >>> df1 = pandas.DataFrame({"id": ["A", "B", "C"], "value": [1, 2, 3]})
                >>> df2 = pandas.DataFrame({"id": ["A", "B", "C"], "value": [1, 2, 3]})
                >>> result = VolumetricMonitor(df1).identifier_comparison(
                ...     df2,
                ...     identifier_columns="id",
                ...     flatten=False
                ... )
                >>> import json
                >>> from pprint import pprint
                >>> pprint(
                ...     result,
                ...     sort_dicts=False
                ... )
                {'volumetrics': [{'test_name': 'Identifier Comparison',
                                  'test_category': 'volumetrics',
                                  'test_type': 'identifier_comparison',
                                  'test_id': 'volumetrics_identifier_comparison',
                                  'values': {'identifiers_match': True,
                                             'dataframe_1': {'identifier_columns': ['id'],
                                                             'record_count': 3,
                                                             'unique_identifier_count': 3,
                                                             'extra_identifiers': {'total': 0,
                                                                                   'breakdown': {}}},
                                             'dataframe_2': {'identifier_columns': ['id'],
                                                             'record_count': 3,
                                                             'unique_identifier_count': 3,
                                                             'extra_identifiers': {'total': 0,
                                                                                   'breakdown': {}}}}}]}

            Compare two dataframes, where the first dataframe has 3 more identifiers (and rows) than
            the second dataframe::

                >>> df1 = pandas.DataFrame(
                ...     {
                ...         "id": ["A", "B", "C", "E", "F", "G"],
                ...         "value": [1, 1, 1, 2, 3, 4]
                ...     }
                ... )

                >>> df2 = pandas.DataFrame(
                ...     {
                ...         "id": ["A", "B", "C"],
                ...         "value": [1, 2, 3]
                ...     }
                ... )

                >>> result = VolumetricMonitor(df1).identifier_comparison(
                ...     df2,
                ...     identifier_columns="id",
                ...     flatten=False
                ... )
                >>> from pprint import pprint
                >>> pprint(
                ...     result,
                ...     sort_dicts=False
                ... )
                {'volumetrics': [{'test_name': 'Identifier Comparison',
                                  'test_category': 'volumetrics',
                                  'test_type': 'identifier_comparison',
                                  'test_id': 'volumetrics_identifier_comparison',
                                  'values': {'identifiers_match': False,
                                             'dataframe_1': {'identifier_columns': ['id'],
                                                             'record_count': 6,
                                                             'unique_identifier_count': 6,
                                                             'extra_identifiers': {'total': 3,
                                                                                   'breakdown': {'E': 1,
                                                                                                 'F': 1,
                                                                                                 'G': 1}}},
                                             'dataframe_2': {'identifier_columns': ['id'],
                                                             'record_count': 3,
                                                             'unique_identifier_count': 3,
                                                             'extra_identifiers': {'total': 0,
                                                                                   'breakdown': {}}}}}]}

            Compare two dataframes using two identifiers: a timestamp and a request id. Note that
            these dataframes have the same value in the `id` column but different timestamps::

                >>> df1 = pandas.DataFrame(
                ...     {
                ...         "id": ["A", "A"],
                ...         "date": ["2021-01-01", "2021-01-01"],
                ...         "value": [1, 2]
                ...     }
                ... )

                >>> df2 = pandas.DataFrame(
                ...     {
                ...         "id": ["A", "A"],
                ...         "date": ["2021-01-01", "2021-02-02"],
                ...         "value": [1, 2]
                ...     }
                ... )

                >>> result = VolumetricMonitor(df1).identifier_comparison(
                ...     df2,
                ...     identifier_columns = ["id", "date"],
                ...     flatten=False
                ... )
                >>> from pprint import pprint
                >>> pprint(
                ...     result,
                ...     sort_dicts=False
                ... )
                {'volumetrics': [{'test_name': 'Identifier Comparison',
                                  'test_category': 'volumetrics',
                                  'test_type': 'identifier_comparison',
                                  'test_id': 'volumetrics_identifier_comparison',
                                  'values': {'identifiers_match': False,
                                             'dataframe_1': {'identifier_columns': ['id',
                                                                                    'date'],
                                                             'record_count': 2,
                                                             'unique_identifier_count': 1,
                                                             'extra_identifiers': {'total': 1,
                                                                                   'breakdown': {'A|2021-01-01': 1}}},
                                             'dataframe_2': {'identifier_columns': ['id',
                                                                                    'date'],
                                                             'record_count': 2,
                                                             'unique_identifier_count': 2,
                                                             'extra_identifiers': {'total': 1,
                                                                                   'breakdown': {'A|2021-02-02': 1}}}}}]}
        """
        # Type check dataframe
        if not isinstance(dataframe_2, pandas.DataFrame):
            raise TypeError("dataframe_2 must be a pandas.DataFrame instance")

        if job_json is not None:
            is_job_json = True
            logger.info(
                "Parameter 'job_json' is present and will be used to extract "
                "'identifier_columns'."
            )

            input_schema_definition = infer.extract_input_schema(job_json)
            monitoring_parameters = infer.set_monitoring_parameters(
                schema_json=input_schema_definition, check_schema=True
            )

            identifier_columns = monitoring_parameters["identifier_columns"]
        else:
            is_job_json = False
            logger.info(
                "Parameter 'job_json' it not present, attempting to use "
                "'identifier_columns' instead."
            )

        if not identifier_columns:
            if is_job_json:
                missing_args_error = "No columns with role=identifier were provided in the input schema! \
                     Please specifiy at least one identifier column in the schema \
                        to be able to use this monitor."
            else:
                missing_args_error = "When passing identifier columns explicitly, identifier_columns cannot be empty."

            logger.error(missing_args_error)
            raise Exception(missing_args_error)

        # Type check identifiers
        if isinstance(identifier_columns, str):
            # Turn single element into list
            identifier_columns = [identifier_columns]
        elif isinstance(identifier_columns, (list, tuple)):
            pass
        else:
            # Invalid type
            raise TypeError(
                "identifier_columns must be string, or a list/tuple of strings."
            )

        # Concatentate identifier columns for each dataframe into a series for each
        dataframe_1_identifiers = self.__df_concat_values(
            self.dataframe, identifier_columns
        )
        dataframe_2_identifiers = self.__df_concat_values(
            dataframe_2, identifier_columns
        )

        # Subtract identifier counts in dataframe_2 from dataframe_1
        extra_on_dataframe_1 = self.__series_except(
            dataframe_1_identifiers, dataframe_2_identifiers
        )
        # Subtract identifier counts in dataframe_1 from dataframe_2
        extra_on_dataframe_2 = self.__series_except(
            dataframe_2_identifiers, dataframe_1_identifiers
        )

        # If there are no extras on either side, then the two dfs match
        identifiers_match = not bool(extra_on_dataframe_1 or extra_on_dataframe_2)

        # Build return structure
        values = {
            "identifiers_match": identifiers_match,
            "dataframe_1": {
                "identifier_columns": identifier_columns,
                "record_count": len(self.dataframe.index),
                "unique_identifier_count": len(dataframe_1_identifiers.unique()),
                "extra_identifiers": {
                    "total": sum(extra_on_dataframe_1.values()),
                    "breakdown": extra_on_dataframe_1,
                },
            },
            "dataframe_2": {
                "identifier_columns": identifier_columns,
                "record_count": len(dataframe_2.index),
                "unique_identifier_count": len(dataframe_2_identifiers.unique()),
                "extra_identifiers": {
                    "total": sum(extra_on_dataframe_2.values()),
                    "breakdown": extra_on_dataframe_2,
                },
            },
        }

        identifier_results = {
            "test_name": "Identifier Comparison",
            "test_category": "volumetrics",
            "test_type": "identifier_comparison",
            "test_id": "volumetrics_identifier_comparison",
            "values": values,
        }

        if flatten:
            result = {
                "identifiers_match": identifier_results["values"]["identifiers_match"],
                result_wrapper_key: [identifier_results],
            }
        else:
            result = {result_wrapper_key: [identifier_results]}

        return result

    @staticmethod
    def __series_except(series1: pandas.Series, series2: pandas.Series) -> dict:
        """
        Returns:
            Dictionary {value: count} of values in `series1` that are missing from `series2`

        Raises:
            TypeError: If `series1` or `series2` is not a pandas Series
        """
        if not isinstance(series1, pandas.Series):
            raise TypeError("series1 must be a pandas.DataFrame instance")
        if not isinstance(series2, pandas.Series):
            raise TypeError("series2 must be a pandas.DataFrame instance")

        # Turn both series into counters of the number of items in each
        counter1 = collections.Counter(series1.value_counts().to_dict())
        counter2 = collections.Counter(series2.value_counts().to_dict())

        # Figure out how many are missing from either side
        missing_values = counter1 - counter2

        # Sort by keys to standardize doctest
        return dict(sorted(missing_values.items()))

    @staticmethod
    def __df_concat_values(
        dataframe: pandas.DataFrame, columns: List, separator="|"
    ) -> pandas.Series:
        """
        Returns:
            Concatentation of `columns` in `dataframe` across each row
        """
        assertions.check_columns_in_dataframe(dataframe, columns)

        # Start with the first column
        combined_series = dataframe[columns[0]].map(str)
        # Add the remaining columns
        for column in columns[1:]:
            combined_series += separator + dataframe[column].map(str)

        return combined_series

def flatten_json(json_to_flatten):
    output = {}

    def flatten(json_symbol, name=''):
        if type(json_symbol) is dict:
            for key in json_symbol:
                flatten(json_symbol[key], name + key + '_')
        elif type(json_symbol) is list:
            i = 0
            for index in json_symbol:
                flatten(index, name + str(i) + '_')
                i += 1
        else:
            output[name[:-1]] = json_symbol

    flatten(json_to_flatten)
    return output


if __name__ == "__main__":
    # in_df = pandas.DataFrame(
    #     {"id": ["A", "A"], "date": ["2021-01-01", "2021-01-01"], "value": [1, 2]}
    # )
    # out_df = pandas.DataFrame(
    #     {"id": ["A", "A"], "date": ["2021-01-01", "2021-02-02"], "value": [1, 2]}
    # )
    # from pprint import pprint

    # pprint(VolumetricMonitor(in_df).identifier_comparison(out_df, ["id", "date"]))

    print(doctest.testmod())
    print()
