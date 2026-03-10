"""This module provides several monitors for data drift (on model inputs) \
and concept drift (on model outputs).

See `ConceptDriftDetector` and `DriftDetector` for usage and examples.

.. toctree::

.. autosummary::
   :toctree: _autosummary
"""

# import copy
import doctest
import logging
import warnings
from typing import List, Optional

# Third party packages
import numpy
import pandas
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from scipy.stats import (
    epps_singleton_2samp,
    gaussian_kde,
    ks_2samp,
    wasserstein_distance,
)

import modelop.schema.infer as infer
from modelop.monitors.assertions import (
    check_baseline_and_sample,
    check_columns_in_dataframe,
    check_date_column,
    check_input_types,
    check_pre_defined_metric,
)
from modelop.utils import get_min_max_values_keys_from_dict

pandas.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


class ConceptDriftDetector:
    """
    Computes differences (by some metric) between a baseline's and a sample's target columns.

    Args:
        df_baseline (pandas.DataFrame): Pandas DataFrame of the baseline dataset.

        df_sample (pandas.DataFrame): Pandas DataFrame of the sample dataset.

        job_json (dict): JSON dictionary with the metadata of the model.

        target_column (str): Column containing predicted (or true) values.

        output_type (str): 'categorical' or 'numerical' to reflect classification or regression.

        date_column (str): Column containing dates for over time metrics.

        decimals (int): Number of decimals to round metrics to.

    Raises:
        AssertionError: If `df_baseline` or `df_sample` are not pandas DataFrames, or their
            column names do not match, or if `target_column` is not among df_baseline.columns,
            or `output_type` is not 'categorical' or 'numerical'.

    Examples:
        Get Jensen-Shannon and Wasserstein distances and KS and ES p-values given two DataFrames:

        >>> import pandas
        >>> from pprint import pprint
        >>> from modelop.monitors.drift import ConceptDriftDetector

        >>> df_1 = pandas.DataFrame(
        ...     [
        ...         {'gender':'male', 'amount':100, 'probability':0.2},
        ...         {'gender':'female', 'amount':200, 'probability':0.9},
        ...         {'gender':'male', 'amount':150, 'probability':0.3},
        ...         {'gender':'male', 'amount':125, 'probability':0.1},
        ...         {'gender':'female', 'amount':175, 'probability':0.8}
        ...     ]
        ... )

        >>> df_2 = pandas.DataFrame(
        ...     [
        ...         {'gender':'female', 'amount':125, 'probability':0.1},
        ...         {'gender':'female', 'amount':150, 'probability':0.7},
        ...         {'gender':'female', 'amount':175, 'probability':0.6},
        ...         {'gender':'male', 'amount':200, 'probability':0.8},
        ...         {'gender':'male', 'amount':190, 'probability':0.4}
        ...     ]
        ... )

        >>> concept_drift_detector=ConceptDriftDetector(
        ...     df_baseline=df_1,
        ...     df_sample=df_2,
        ...     target_column='probability',
        ...     output_type='numerical',
        ...     decimals=3,
        ... )

        >>> pprint(
        ...     concept_drift_detector.calculate_concept_drift(
        ...         pre_defined_test='js',
        ...         result_wrapper_key=None,
        ...         include_min_max_features=False
        ...     ),
        ...     sort_dicts=False
        ... )
        {'concept_drift': [{'test_name': 'Jensen-Shannon',
                            'test_category': 'concept_drift',
                            'test_type': 'jensen_shannon',
                            'metric': 'distance',
                            'test_id': 'concept_drift_jensen_shannon_distance',
                            'values': {'probability': 0.118}}]}

        >>> pprint(
        ...     concept_drift_detector.calculate_concept_drift(
        ...         pre_defined_test='wasserstein',
        ...         result_wrapper_key=None,
        ...         include_min_max_features=False
        ...     ),
        ...     sort_dicts=False
        ... )
        {'concept_drift': [{'test_name': 'Wasserstein',
                            'test_category': 'concept_drift',
                            'test_type': 'wasserstein',
                            'metric': 'distance',
                            'test_id': 'concept_drift_wasserstein_distance',
                            'values': {'probability': 0.14}}]}

        >>> pprint(
        ...     concept_drift_detector.calculate_concept_drift(
        ...         pre_defined_test='kolmogorov-Smirnov',
        ...         result_wrapper_key=None,
        ...         include_min_max_features=False
        ...     ),
        ...     sort_dicts=False
        ... )
        {'concept_drift': [{'test_name': 'Kolmogorov-Smirnov',
                            'test_category': 'concept_drift',
                            'test_type': 'kolmogorov_smirnov',
                            'metric': 'p_value',
                            'test_id': 'concept_drift_kolmogorov_smirnov_p_value',
                            'values': {'probability': 0.873}}]}

        >>> pprint(
        ...     concept_drift_detector.calculate_concept_drift(
        ...         pre_defined_test='EPPS-SINGLETON',
        ...         result_wrapper_key=None,
        ...         include_min_max_features=False
        ...     ),
        ...     sort_dicts=False
        ... )
        {'concept_drift': [{'test_name': 'Epps-Singleton',
                            'test_category': 'concept_drift',
                            'test_type': 'epps_singleton',
                            'metric': 'p_value',
                            'test_id': 'concept_drift_epps_singleton_p_value',
                            'values': {'probability': 0.38}}]}

        Get KS and ES statistics instead of p-values:

        >>> pprint(
        ...     concept_drift_detector.calculate_concept_drift(
        ...         pre_defined_test='KS', metric='statistic',
        ...         result_wrapper_key=None,
        ...         include_min_max_features=False
        ...     ),
        ...     sort_dicts=False
        ... )
        {'concept_drift': [{'test_name': 'Kolmogorov-Smirnov',
                            'test_category': 'concept_drift',
                            'test_type': 'kolmogorov_smirnov',
                            'metric': 'statistic',
                            'test_id': 'concept_drift_kolmogorov_smirnov_statistic',
                            'values': {'probability': 0.4}}]}

        >>> pprint(
        ...     concept_drift_detector.calculate_concept_drift(
        ...         pre_defined_test='ES', metric='statistic',
        ...         result_wrapper_key=None,
        ...         include_min_max_features=False
        ...     ),
        ...     sort_dicts=False
        ... )
        {'concept_drift': [{'test_name': 'Epps-Singleton',
                            'test_category': 'concept_drift',
                            'test_type': 'epps_singleton',
                            'metric': 'statistic',
                            'test_id': 'concept_drift_epps_singleton_statistic',
                            'values': {'probability': 4.199}}]}

        Compute KL divergence:

        >>> pprint(
        ...     concept_drift_detector.calculate_concept_drift(
        ...         pre_defined_test='Kullback-Leibler',
        ...         num_buckets=3,
        ...         result_wrapper_key=None,
        ...         include_min_max_features=False
        ...     ),
        ...     sort_dicts=False
        ... )
        {'concept_drift': [{'test_name': 'Kullback-Leibler',
                            'test_category': 'concept_drift',
                            'test_type': 'kullback_leibler',
                            'metric': 'divergence',
                            'test_id': 'concept_drift_kullback_leibler_divergence',
                            'values': {'probability': 0.659}}]}
    """

    def __init__(
        self,
        df_baseline: pandas.DataFrame,
        df_sample: pandas.DataFrame,
        job_json: dict = None,
        target_column: str = None,
        output_type: str = None,
        date_column: str = None,
        decimals: Optional[int] = 4,
    ) -> None:

        check_baseline_and_sample(
            df_baseline=df_baseline, df_sample=df_sample, check_column_equality=False
        )

        if job_json is not None:
            logger.info(
                "Parameter 'job_json' is present and will be used to extract "
                "'target_column' and 'output_type'."
            )
            input_schema_definition = infer.extract_input_schema(job_json)
            monitoring_parameters = infer.set_monitoring_parameters(
                schema_json=input_schema_definition, check_schema=True
            )

            target_column = monitoring_parameters["score_column"]
            output_type = monitoring_parameters["output_type"]
            date_column = monitoring_parameters["date_column"]
        else:
            logger.info(
                "Parameter 'job_json' it not present, attempting to use "
                "'target_column' and 'output_type' instead."
            )
            if target_column is None or output_type is None:
                missing_args_error = (
                    "Parameter 'job_json' is not present,"
                    " but one of 'target_column' and 'output_type' was not provided. "
                    "Both 'target_column' and 'output_type' input parameters are"
                    " required if 'job_json' is not provided."
                )
                logger.error(missing_args_error)
                raise Exception(missing_args_error)

        check_columns_in_dataframe(dataframe=df_baseline, columns=[target_column])

        if date_column is not None:
            check_columns_in_dataframe(
                dataframe=df_sample, columns=[target_column, date_column]
            )
            logger.info("Restricting dataframe to columns %s", [target_column, date_column])
            self.df_baseline = df_baseline[[target_column]]
            self.df_sample = df_sample[[target_column, date_column]]
        else:
            logger.info("Restricting dataframe to columns %s", [target_column])
            self.df_baseline = df_baseline[[target_column]]
            self.df_sample = df_sample[[target_column]]

        if date_column is not None:
            check_input_types(inputs=[{"date_column": date_column}], types=(str))

        assert output_type in (
            "categorical",
            "numerical",
        ), "output_type should be either 'categorical' or 'numerical'."

        # Set attributes
        self.target_column = target_column
        self.date_column = date_column
        self.output_type = output_type
        self.decimals = decimals

        if self.output_type == "categorical":
            self.categorical_columns = [self.target_column]
            self.numerical_columns = []
        elif self.output_type == "numerical":
            self.categorical_columns = []
            self.numerical_columns = [self.target_column]

        # TODO:
        # - Include field for special values, like num_buckets, for specific tests

    def __str__(self):
        return self.__class__.__name__

    def calculate_concept_drift(
        self,
        pre_defined_test: str = None,
        flattening_suffix: Optional[str] = None,
        result_wrapper_key: Optional[str] = "concept_drift",
        include_min_max_features: bool = True,
        include_over_time: bool = True,
        **kwargs,
    ):
        """
        Calculates concept drift between target columns of baseline and sample datasets
        according to a pre-defined metric or a user-defined metric.

        Args:
            pre_defined_test (str): 'jensen-shannon' ('js'),
                or 'kolmogorov-smirnov' ('ks'),
                or 'epps-singleton' ('es'),
                or 'wasserstein' ('ws'),
                or 'kullback-leibler' ('kl'),
                or 'summary' ('describe').

            flattening_suffix (str): If defined, provide a flattened output with the given suffix.

            result_wrapper_key (str): Provides a key to wrap the result dict (flattened results are outside this entry).

            include_min_max_features (bool): Provides the min and max values of the flattened results,
                and their corresponding feature names.
            
            include_over_time (bool): Includes a new section with metrics calculation over a prediction date column.

        Returns:
            Drift measures as computed by some metrics function.
        """

        # Remove capitalization
        pre_defined_test = pre_defined_test.lower()

        # Make sure choice is valid
        check_pre_defined_metric(
            pre_defined_test=pre_defined_test,
            metrics_function="calculate_concept_drift",
        )

        values_over_time = {}

        if pre_defined_test in ["jensen-shannon", "js"]:
            logger.info("Computing JS metrics")
            test_name = "Jensen-Shannon"

            metric = "distance"

            values = js_metric(
                df_1=self.df_baseline,
                df_2=self.df_sample,
                numerical_columns=self.numerical_columns,
                categorical_columns=self.categorical_columns,
                decimals=self.decimals,
            )
            if include_over_time and self.date_column is not None:
                values_over_time = self.concept_drift_over_time(js_metric, [], result_wrapper_key, test_name, {"numerical_columns": self.numerical_columns,
                                                                                                               "categorical_columns": self.categorical_columns})
        elif pre_defined_test in ["kolmogorov-smirnov", "ks"]:
            logger.info("Computing KS metrics")
            # TODO:
            # assert (
            #     self.output_type == "numerical"
            # ), "KS metric not available for categorical labels."
            test_name = "Kolmogorov-Smirnov"

            metric = kwargs["metric"] if "metric" in kwargs.keys() else "p-value"
            assert metric.lower() in [
                "p-value",
                "statistic",
            ], "'metric' should be one of ['p-value', 'statistic']."

            values = ks_metric(
                df_1=self.df_baseline,
                df_2=self.df_sample,
                numerical_columns=[self.target_column],
                metric=metric.lower(),
                decimals=self.decimals,
            )

            if include_over_time and self.date_column is not None:
                values_over_time = self.concept_drift_over_time(ks_metric, [], result_wrapper_key, test_name, {"numerical_columns": [self.target_column],
                                                                                                               "metric": metric.lower()})
        elif pre_defined_test in ["epps-singleton", "es"]:
            logger.info("Computing ES metrics")
            # TODO:
            # assert (
            #     self.output_type == "numerical"
            # ), "ES metric not available for categorical labels."
            test_name = "Epps-Singleton"

            metric = kwargs["metric"] if "metric" in kwargs.keys() else "p-value"
            assert metric.lower() in [
                "p-value",
                "statistic",
            ], "'metric' should be one of ['p-value', 'statistic']."

            values = es_metric(
                df_1=self.df_baseline,
                df_2=self.df_sample,
                numerical_columns=[self.target_column],
                metric=metric.lower(),
                decimals=self.decimals,
            )

            if include_over_time and self.date_column is not None:
                values_over_time = self.concept_drift_over_time(es_metric, [], result_wrapper_key, test_name, {"numerical_columns": [self.target_column],
                                                                                                               "metric": metric.lower()})
        elif pre_defined_test in ["kullback-leibler", "kl"]:
            logger.info("Computing KL metrics")

            test_name = "Kullback-Leibler"

            metric = "divergence"

            if self.output_type == "numerical":
                # Perhaps move this to init function
                if kwargs.get("num_buckets", None) == None:
                    num_buckets = 5
                else:
                    num_buckets = kwargs["num_buckets"]

                values = kl_metric(
                    df_1=self.df_baseline,
                    df_2=self.df_sample,
                    numerical_columns=[self.target_column],
                    categorical_columns=[],
                    num_buckets=num_buckets,
                    decimals=self.decimals,
                )
                if include_over_time and self.date_column is not None:
                    values_over_time = self.concept_drift_over_time(kl_metric, [], result_wrapper_key, test_name, {"numerical_columns": [self.target_column],
                                                                                                                   "categorical_columns": [],
                                                                                                                   "num_buckets": num_buckets})
            elif self.output_type == "categorical":
                values = kl_metric(
                    df_1=self.df_baseline,
                    df_2=self.df_sample,
                    categorical_columns=[self.target_column],
                    numerical_columns=[],
                    num_buckets=0,
                    decimals=self.decimals,
                )
                if include_over_time and self.date_column is not None:
                    values_over_time = self.concept_drift_over_time(kl_metric, [], result_wrapper_key, test_name, {"numerical_columns": [],
                                                                                                                   "categorical_columns": [self.target_column],
                                                                                                                   "num_buckets": 0})
        elif pre_defined_test in ["wasserstein", "ws"]:
            logger.info("Computing WS metrics")
            # assert (
            #     self.output_type == "numerical"
            # ), "WS metric not available for categorical labels"

            test_name = "Wasserstein"

            metric = "distance"

            values = ws_metric(
                df_1=self.df_baseline,
                df_2=self.df_sample,
                numerical_columns=[self.target_column],
                decimals=self.decimals,
            )

            if include_over_time and self.date_column is not None:
                values_over_time = self.concept_drift_over_time(ws_metric, [], result_wrapper_key, test_name, {"numerical_columns": [self.target_column]})
        elif pre_defined_test in ["describe", "summary"]:
            logger.info("Computing Summary metrics")
            test_name = "Summary"

            metric = "pandas_describe"

            values = compare_dataframe_columns(
                df_1=self.df_baseline,
                df_2=self.df_sample,
                numerical_columns=self.numerical_columns,
                categorical_columns=self.categorical_columns,
                decimals=self.decimals,
            )

        test_type = test_name.lower().replace("-", "_")
        metric = metric.lower().replace("-", "_")
        concept_drift_result = {
            "test_name": test_name,
            "test_category": "concept_drift",
            "test_type": test_type,
            "metric": metric,
            "test_id": "concept_drift_{}_{}".format(test_type, metric),
            "values": values,
        }

        result = {}
        if include_over_time and self.date_column is not None:
            if flattening_suffix is not None:
                values_over_time = {
                    "concept_drift_over_time" + flattening_suffix: values_over_time["concept_drift_over_time"],
                    "lastPredictionDate": values_over_time["lastPredictionDate"],
                    "firstPredictionDate": values_over_time["firstPredictionDate"]
                }
                result.update(values_over_time)
            else:
                result.update(values_over_time)

        if flattening_suffix is not None:
            result.update({
                # Top-level metrics
                str(feature + flattening_suffix): concept_drift_result["values"][
                    feature
                ]
                for feature in concept_drift_result["values"].keys()
            })

        # Include min and max metric values
        if include_min_max_features and test_name != "Summary":

            min_max_dict = get_min_max_values_keys_from_dict(
                values_dict=concept_drift_result["values"]
            )

            metric_camel = "".join([i.capitalize() for i in metric.split("_")])
            result[f"ConceptDrift_max{test_name}{metric_camel}Value"] = min_max_dict[
                "max_value"
            ]
            result[
                f"ConceptDrift_max{test_name}{metric_camel}ValueFeature"
            ] = min_max_dict["max_feature"]
            result[f"ConceptDrift_min{test_name}{metric_camel}Value"] = min_max_dict[
                "min_value"
            ]
            result[
                f"ConceptDrift_min{test_name}{metric_camel}ValueFeature"
            ] = min_max_dict["min_feature"]

        # Add Vanilla ConceptDriftDetector output
        if result_wrapper_key is not None:
            result[result_wrapper_key] = [concept_drift_result]
        else:
            result["concept_drift"] = [concept_drift_result]
        return result

    def concept_drift_over_time(self, evaluation_function,
                              exclude_metrics: list = (),
                              result_wrapper_key: str = 'concept_drift',
                              test_name: str = '',
                              custom_args: dict = {}
                              ) -> dict:
        """
        Computes the concept drift metrics as given by the evaluation_function, attempting to split the data by date
        :param evaluation_function: The function to run per data split (by date)
        :param exclude_metrics: A list of keys to exclude from the result of the evaluation_function
        :param result_wrapper_key: The key used for the wrapping the whole test results. This will create a similar key
        :param test_name: The name of the specific test actually run.
        :param custom_args: A dict of additional kwargs to pass to the evaluation_function.
        :return: A dictionary with a graph structure over time
        """
        if self.date_column is not None and self.date_column in self.df_sample:
            df2 = self.date_index_dataframe(self.df_sample)

            unique_dates = df2[self.date_column].unique()

            data = {}
            for date in unique_dates:
                data_of_the_day_sample = df2.loc[[date]]
                dated_values = evaluation_function(df_1=self.df_baseline, df_2=data_of_the_day_sample, decimals=self.decimals, **custom_args)

                str_date = str(date)
                for metric in dated_values:
                    if metric in exclude_metrics:
                        continue
                    if metric not in data:
                        data[metric] = []
                    data[metric].append([str_date, dated_values[metric]])

            over_time_key = result_wrapper_key + "_over_time"
            return {over_time_key: {
                "title": "Concept Drift Over Time" + (" - " + test_name if test_name else ""),
                "x_axis_label": "Day",
                "y_axis_label": "Metric",
                "data": data
            },
                "firstPredictionDate": str(unique_dates.min()),
                "lastPredictionDate": str(unique_dates.max()),
            }
        else:
            return {}

    def date_index_dataframe(self, df: pandas.DataFrame):
        """
        Converts the dates in the date_column to pandas dates, and indexes the dataframe by the dates.
        :param df: The dataframe to index by dates
        :return: The indexed and sorted dataframe
        """
        df = df.set_index(check_date_column(df, self.date_column).dt.date)
        df[self.date_column] = check_date_column(df, self.date_column).dt.date
        df = df.sort_index()
        return df

class DriftDetector:
    """
    Computes differences (by some metric) between two DataFrames: a baseline and a sample.

    Args:
        df_baseline (pandas.DataFrame): Pandas DataFrame of the baseline dataset.

        df_sample (pandas.DataFrame): Pandas DataFrame of the sample dataset.

        job_json (dict): JSON dictionary with the metadata of the model.

        categorical_columns (List[str]): A list of categorical columns in the dataset.
            If not provided, categorical columns will be inferred from column types.

        numerical_columns (List[str]): A list of numerical columns in the dataset.
            If not provided, numerical columns will be inferred from column types.

        date_column (str): Column containing dates for over time metrics.

        decimals (int): Number of decimals to round metrics to.

    Raises:
        AssertionError: If `df_baseline` or `df_sample` are not pandas DataFrames,
            or their column names do not match, or if `categorical_columns` or
            `numerical_columns` are not lists.

    Examples:
        Get Jensen-Shannon and Wasserstein distances and KS and ES p-values given two DataFrames:

        >>> import pandas
        >>> from pprint import pprint
        >>> from modelop.monitors.drift import DriftDetector

        >>> df_1 = pandas.DataFrame(
        ...     [
        ...         {'gender':'male', 'amount':100, 'label':0},
        ...         {'gender':'female', 'amount':200, 'label':1},
        ...         {'gender':'male', 'amount':150, 'label':0},
        ...         {'gender':'male', 'amount':125, 'label':0},
        ...         {'gender':'female', 'amount':175, 'label':1}
        ...     ]
        ... )

        >>> df_2 = pandas.DataFrame(
        ...     [
        ...         {'gender':'female', 'amount':125, 'label':0},
        ...         {'gender':'female', 'amount':150, 'label':1},
        ...         {'gender':'female', 'amount':175, 'label':1},
        ...         {'gender':'male', 'amount':200, 'label':1},
        ...         {'gender':'male', 'amount':190, 'label':0}
        ...     ]
        ... )

        >>> drift_detector=DriftDetector(
        ...     df_baseline=df_1,
        ...     df_sample=df_2,
        ...     categorical_columns=['gender'],
        ...     numerical_columns=['amount'],
        ... )

        >>> pprint(
        ...     drift_detector.calculate_drift(
        ...         pre_defined_test='js',
        ...         result_wrapper_key=None,
        ...         include_min_max_features=True,
        ...         flattening_suffix='_js_distance',
        ...     ),
        ...     sort_dicts=False,
        ... )
        {'gender_js_distance': 0.1419,
         'amount_js_distance': 0.1215,
         'DataDrift_maxJensen-ShannonDistance': 0.1419,
         'DataDrift_maxJensen-ShannonDistanceFeature': 'gender',
         'DataDrift_minJensen-ShannonDistance': 0.1215,
         'DataDrift_minJensen-ShannonDistanceFeature': 'amount',
         'data_drift': [{'test_name': 'Jensen-Shannon',
                         'test_category': 'data_drift',
                         'test_type': 'jensen_shannon',
                         'metric': 'distance',
                         'test_id': 'data_drift_jensen_shannon_distance',
                         'values': {'gender': 0.1419, 'amount': 0.1215}}]}

        >>> pprint(
        ...     drift_detector.calculate_drift(
        ...         pre_defined_test='ws',
        ...         result_wrapper_key=None,
        ...         include_min_max_features=False
        ...     ),
        ...     sort_dicts=False,
        ... )
        {'data_drift': [{'test_name': 'Wasserstein',
                         'test_category': 'data_drift',
                         'test_type': 'wasserstein',
                         'metric': 'distance',
                         'test_id': 'data_drift_wasserstein_distance',
                         'values': {'amount': 18.0}}]}

        >>> pprint(
        ...     drift_detector.calculate_drift(
        ...         pre_defined_test='Kolmogorov-Smirnov',
        ...         result_wrapper_key='DATA_DRIFT',
        ...         include_min_max_features=False,
        ...         flattening_suffix='_ks_pvalue',
        ...     ),
        ...     sort_dicts=False,
        ... )
        {'amount_ks_pvalue': 1.0,
         'DATA_DRIFT': [{'test_name': 'Kolmogorov-Smirnov',
                         'test_category': 'data_drift',
                         'test_type': 'kolmogorov_smirnov',
                         'metric': 'p_value',
                         'test_id': 'data_drift_kolmogorov_smirnov_p_value',
                         'values': {'amount': 1.0}}]}

        >>> pprint(
        ...     drift_detector.calculate_drift(
        ...         pre_defined_test='EPPS-SINGLETON',
        ...         result_wrapper_key=None,
        ...         include_min_max_features=True
        ...     ),
        ...     sort_dicts=False,
        ... )
        {'DataDrift_maxEpps-SingletonPValue': 0.9579,
         'DataDrift_maxEpps-SingletonPValueFeature': 'amount',
         'DataDrift_minEpps-SingletonPValue': 0.9579,
         'DataDrift_minEpps-SingletonPValueFeature': 'amount',
         'data_drift': [{'test_name': 'Epps-Singleton',
                         'test_category': 'data_drift',
                         'test_type': 'epps_singleton',
                         'metric': 'p_value',
                         'test_id': 'data_drift_epps_singleton_p_value',
                         'values': {'amount': 0.9579}}]}

        Get KS and ES statistics instead of p-values:

        >>> pprint(
        ...     drift_detector.calculate_drift(
        ...         pre_defined_test='KS',
        ...         metric='statistic',
        ...         result_wrapper_key=None,
        ...         include_min_max_features=False
        ...     ),
        ...     sort_dicts=False,
        ... )
        {'data_drift': [{'test_name': 'Kolmogorov-Smirnov',
                         'test_category': 'data_drift',
                         'test_type': 'kolmogorov_smirnov',
                         'metric': 'statistic',
                         'test_id': 'data_drift_kolmogorov_smirnov_statistic',
                         'values': {'amount': 0.2}}]}

        >>> pprint(
        ...     drift_detector.calculate_drift(
        ...         pre_defined_test='ES',
        ...         metric='statistic',
        ...         result_wrapper_key=None,
        ...         include_min_max_features=False
        ...     ),
        ...     sort_dicts=False,
        ... )
        {'data_drift': [{'test_name': 'Epps-Singleton',
                         'test_category': 'data_drift',
                         'test_type': 'epps_singleton',
                         'metric': 'statistic',
                         'test_id': 'data_drift_epps_singleton_statistic',
                         'values': {'amount': 0.645}}]}

        Compute KL divergence:

        >>> pprint(
        ...     drift_detector.calculate_drift(
        ...         pre_defined_test='KL',
        ...         num_buckets=5,
        ...         result_wrapper_key=None,
        ...         include_min_max_features=False
        ...     ),
        ...     sort_dicts=False,
        ... )
        {'data_drift': [{'test_name': 'Kullback-Leibler',
                         'test_category': 'data_drift',
                         'test_type': 'kullback_leibler',
                         'metric': 'divergence',
                         'test_id': 'data_drift_kullback_leibler_divergence',
                         'values': {'amount': 0.2773, 'gender': 0.0811}}]}
    """

    def __init__(
        self,
        df_baseline: pandas.DataFrame,
        df_sample: pandas.DataFrame,
        job_json: dict = None,
        categorical_columns: Optional[list] = None,
        numerical_columns: Optional[list] = None,
        date_column: str = None,
        decimals: Optional[int] = 4,
    ) -> None:

        check_baseline_and_sample(
            df_baseline=df_baseline, df_sample=df_sample, check_column_equality=False
        )

        if job_json is not None:
            logger.info(
                "Parameter 'job_json' is present and will be used to extract numerical "
                "and categorical columns from it."
            )
            input_schema_definition = infer.extract_input_schema(job_json)
            monitoring_parameters = infer.set_monitoring_parameters(
                schema_json=input_schema_definition, check_schema=True
            )
            numerical_columns = monitoring_parameters["numerical_columns"]
            categorical_columns = monitoring_parameters["categorical_columns"]
            date_column = monitoring_parameters["date_column"]
        else:
            logger.info(
                "Parameter 'job_json' is not present, "
                "attempting to use 'numerical_columns' and 'categorical_columns' instead."
            )
            if categorical_columns is None and numerical_columns is None:
                missing_args_error = (
                    "Parameter 'job_json' is not present,"
                    " but neither 'categorical_columns' or 'numerical_columns' was not provided. "
                    "At least one of 'numerical_columns' and 'categorical_columns' input parameters are required"
                    " if 'job_json' is not provided."
                )
                logger.error(missing_args_error)
                raise Exception(missing_args_error)

        check_input_types(
            inputs=[
                {"categorical_columns": categorical_columns},
                {"numerical_columns": numerical_columns},
            ],
            types=(list, type(None)),
        )

        if categorical_columns is not None:
            check_columns_in_dataframe(
                dataframe=df_baseline, columns=categorical_columns
            )
        if numerical_columns is not None:
            check_columns_in_dataframe(dataframe=df_baseline, columns=numerical_columns)

        if date_column is not None:
            check_columns_in_dataframe(dataframe=df_sample, columns=[date_column])

        if date_column is not None:
            check_input_types(inputs=[{"date_column": date_column}], types=(str))

        # df_baseline_ = copy.deepcopy(df_baseline)
        # df_sample_ = copy.deepcopy(df_sample)

        # Infer categorical columns if not specified
        if categorical_columns is None:
            categorical_columns = [
                col
                for col in df_baseline.columns
                if df_baseline.dtypes[col] == "object"
            ]
            print("Identified categorical column(s): ", categorical_columns)
        else:
            categorical_columns = list(categorical_columns)

        # Cast categorical values as strings (list indexer required by pandas)
        if categorical_columns:
            df_baseline[categorical_columns] = df_baseline[categorical_columns].astype(str)
            df_sample[categorical_columns] = df_sample[categorical_columns].astype(str)

        # Infer numerical columns if not specified
        if numerical_columns is None:
            num_types = ["float64", "float32", "int32", "int64", "uint8"]
            numerical_columns = [
                col
                for col in df_baseline.columns
                if df_baseline.dtypes[col] in num_types
            ]
            print("Identified numerical column(s): ", numerical_columns)
        else:
            numerical_columns = list(numerical_columns)

        # Cast numerical values as floats (list indexer required by pandas)
        if numerical_columns:
            df_baseline[numerical_columns] = df_baseline[numerical_columns].astype(float)
            df_sample[numerical_columns] = df_sample[numerical_columns].astype(float)

        # Set attributes
        self.df_baseline = df_baseline
        self.df_sample = df_sample

        self.date_column = date_column
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        self.decimals = decimals

        # TODO:
        # - Include field for special values, like num_buckets, for specific tests

    def __str__(self):
        return self.__class__.__name__

    def calculate_drift(
        self,
        pre_defined_test: str = None,
        flattening_suffix: Optional[str] = None,
        result_wrapper_key: Optional[str] = "data_drift",
        include_min_max_features: bool = True,
        include_over_time: bool = True,
        **kwargs,
    ):
        """
        Calculates drift between baseline and sample datasets according to a
        pre-defined metric or a user-defined metric.

        Args:
            pre_defined_test (str): 'jensen-shannon' ('js'),
                or 'kolmogorov-smirnov' ('ks'),
                or 'epps-singleton' ('es'),
                or 'kullback-leibler' ('kl'),
                or 'summary' ('describe').

            flattening_suffix (str): If defined, provide a flattened output with the given suffix.

            result_wrapper_key (str): Provides a key to wrap the result dict (flattened results are outside this entry).

            include_min_max_features (bool): Provides the min and max values of the flattened results,
                and their corresponding feature names.

            include_over_time (bool): Includes a new section with metrics calculation over a prediction date column.

        Returns:
            Drift measures as computed by some metrics function.
        """

        # Remove capitalization
        pre_defined_test = pre_defined_test.lower()

        # Make sure choice is valid
        check_pre_defined_metric(
            pre_defined_test=pre_defined_test,
            metrics_function="calculate_drift",
        )

        values_over_time = {}

        if pre_defined_test in ["jensen-shannon", "js"]:
            test_name = "Jensen-Shannon"

            metric = "distance"

            values = js_metric(
                df_1=self.df_baseline,
                df_2=self.df_sample,
                numerical_columns=self.numerical_columns,
                categorical_columns=self.categorical_columns,
                decimals=self.decimals,
            )

            if include_over_time and self.date_column is not None:
                values_over_time = self.data_drift_over_time(js_metric, [], result_wrapper_key, test_name, {"numerical_columns": self.numerical_columns,
                                                                                                            "categorical_columns": self.categorical_columns})
        elif pre_defined_test in ["wasserstein", "ws"]:
            test_name = "Wasserstein"

            metric = "distance"

            values = ws_metric(
                df_1=self.df_baseline,
                df_2=self.df_sample,
                numerical_columns=self.numerical_columns,
                decimals=self.decimals,
            )

            if include_over_time and self.date_column is not None:
                values_over_time = self.data_drift_over_time(ws_metric, [], result_wrapper_key, test_name, {"numerical_columns": self.numerical_columns})
        elif pre_defined_test in ["kolmogorov-smirnov", "ks"]:
            test_name = "Kolmogorov-Smirnov"

            metric = kwargs["metric"] if "metric" in kwargs.keys() else "p-value"
            assert metric.lower() in [
                "p-value",
                "statistic",
            ], "'metric' should be one of ['p-value', 'statistic']."

            values = ks_metric(
                df_1=self.df_baseline,
                df_2=self.df_sample,
                numerical_columns=self.numerical_columns,
                metric=metric.lower(),
                decimals=self.decimals,
            )

            if include_over_time and self.date_column is not None:
                values_over_time = self.data_drift_over_time(ks_metric, [], result_wrapper_key, test_name, {"numerical_columns": self.numerical_columns,
                                                                                                            "metric": metric.lower()})
        elif pre_defined_test in ["epps-singleton", "es"]:
            test_name = "Epps-Singleton"

            metric = kwargs["metric"] if "metric" in kwargs.keys() else "p-value"
            assert metric.lower() in [
                "p-value",
                "statistic",
            ], "'metric' should be one of ['p-value', 'statistic']."

            values = es_metric(
                df_1=self.df_baseline,
                df_2=self.df_sample,
                numerical_columns=self.numerical_columns,
                metric=metric.lower(),
                decimals=self.decimals,
            )

            if include_over_time and self.date_column is not None:
                values_over_time = self.data_drift_over_time(es_metric, [], result_wrapper_key, test_name, {"numerical_columns": self.numerical_columns,
                                                                                                            "metric": metric.lower()})
        elif pre_defined_test in ["kullback-leibler", "kl"]:
            test_name = "Kullback-Leibler"

            metric = "divergence"

            # Perhaps move this to init function
            if kwargs.get("num_buckets", None) == None:
                num_buckets = 5
            else:
                num_buckets = kwargs["num_buckets"]

            values = kl_metric(
                df_1=self.df_baseline,
                df_2=self.df_sample,
                numerical_columns=self.numerical_columns,
                categorical_columns=self.categorical_columns,
                num_buckets=num_buckets,
                decimals=self.decimals,
            )

            if include_over_time and self.date_column is not None:
                values_over_time = self.data_drift_over_time(kl_metric, [], result_wrapper_key, test_name, {"numerical_columns": self.numerical_columns,
                                                                                                            "categorical_columns": self.categorical_columns,
                                                                                                            "num_buckets": num_buckets})
        elif pre_defined_test in ["describe", "summary"]:
            test_name = "Summary"

            metric = "pandas_describe"

            values = compare_dataframe_columns(
                df_1=self.df_baseline,
                df_2=self.df_sample,
                numerical_columns=self.numerical_columns,
                categorical_columns=self.categorical_columns,
                decimals=self.decimals,
            )

        test_type = test_name.lower().replace("-", "_")
        metric = metric.lower().replace("-", "_")
        drift_result = {
            "test_name": test_name,
            "test_category": "data_drift",
            "test_type": test_type,
            "metric": metric,
            "test_id": "data_drift_{}_{}".format(test_type, metric),
            "values": values,
        }
        result = {}

        if include_over_time and self.date_column is not None:
            if flattening_suffix is not None:
                values_over_time = {
                    "data_drift_over_time" + flattening_suffix: values_over_time["data_drift_over_time"],
                    "lastPredictionDate": values_over_time["lastPredictionDate"],
                    "firstPredictionDate": values_over_time["firstPredictionDate"]
                }
                result.update(values_over_time)
            else:
                result.update(values_over_time)

        if flattening_suffix is not None:
            result.update({
                # Top-level metrics
                str(feature + flattening_suffix): drift_result["values"][feature]
                for feature in drift_result["values"].keys()
            })

        # Include min and max metric values
        if include_min_max_features and test_name != "Summary":

            min_max_dict = get_min_max_values_keys_from_dict(
                values_dict=drift_result["values"]
            )

            metric_camel = "".join([i.capitalize() for i in metric.split("_")])
            result[f"DataDrift_max{test_name}{metric_camel}"] = min_max_dict[
                "max_value"
            ]
            result[f"DataDrift_max{test_name}{metric_camel}Feature"] = min_max_dict[
                "max_feature"
            ]
            result[f"DataDrift_min{test_name}{metric_camel}"] = min_max_dict[
                "min_value"
            ]
            result[f"DataDrift_min{test_name}{metric_camel}Feature"] = min_max_dict[
                "min_feature"
            ]

        # Add Vanilla DriftDetector output
        if result_wrapper_key is not None:
            result[result_wrapper_key] = [drift_result]
        else:
            result["data_drift"] = [drift_result]
        return result

    def data_drift_over_time(self, evaluation_function,
                              exclude_metrics: list = (),
                              result_wrapper_key: str = 'data_drift',
                              test_name: str = '',
                              custom_args: dict = {}
                              ) -> dict:
        """
        Computes the data drift metrics as given by the evaluation_function, attempting to split the data by date
        :param evaluation_function: The function to run per data split (by date)
        :param exclude_metrics: A list of keys to exclude from the result of the evaluation_function
        :param result_wrapper_key: The key used for the wrapping the whole test results. This will create a similar key
        :param test_name: The name of the specific test actually run.
        :param custom_args: A dict of additional kwargs to pass to the evaluation_function.
        :return: A dictionary with a graph structure over time
        """
        if self.date_column is not None and self.date_column in self.df_sample:
            df2 = self.date_index_dataframe(self.df_sample)

            unique_dates = df2[self.date_column].unique()

            data = {}
            for date in unique_dates:
                data_of_the_day_sample = df2.loc[[date]]
                dated_values = evaluation_function(df_1=self.df_baseline, df_2=data_of_the_day_sample, decimals=self.decimals, **custom_args)

                str_date = str(date)
                for metric in dated_values:
                    if metric in exclude_metrics:
                        continue
                    if metric not in data:
                        data[metric] = []
                    data[metric].append([str_date, dated_values[metric]])

            over_time_key = result_wrapper_key + "_over_time"
            return {over_time_key: {
                "title": "Data Drift Over Time" + (" - " + test_name if test_name else ""),
                "x_axis_label": "Day",
                "y_axis_label": "Metric",
                "data": data
            },
                "firstPredictionDate": str(unique_dates.min()),
                "lastPredictionDate": str(unique_dates.max()),
            }
        else:
            return {}

    def date_index_dataframe(self, df: pandas.DataFrame):
        """
        Converts the dates in the date_column to pandas dates, and indexes the dataframe by the dates.
        :param df: The dataframe to index by dates
        :return: The indexed and sorted dataframe
        """
        df = df.set_index(check_date_column(df, self.date_column).dt.date)
        df[self.date_column] = check_date_column(df, self.date_column).dt.date
        df = df.sort_index()
        return df


def ks_metric(
    df_1: pandas.DataFrame,
    df_2: pandas.DataFrame,
    numerical_columns: List[str],
    metric: Optional[str] = "p-value",
    decimals: Optional[int] = 4,
) -> dict:
    """
    Computes p_values corresponding to the KS (Kolmogorov-Smirnov) two_sample statistic.

    Args:
        df_1 (pandas.DataFrame): baseline DataFrame.

        df_2 (pandas.DataFrame): sample DataFrame.

        numerical_columns (List[str]): list of numerical columns to compare.

        metric (str): Metric to return ('p-value' or 'statistic'). Default is 'p-value'.

        decimals (int): Number of decimals to round metrics to.

    Returns:
        Dictionary of p_values or statistics, one for each column in numerical_columns.
    """

    ks_tests = []
    for feat in numerical_columns:
        logger.info("Computing KS on numerical_column %s", feat)
        ks_tests.append(ks_2samp(data1=df_1.loc[:, feat], data2=df_2.loc[:, feat]))

    pvalues = [x[1].round(decimals) for x in ks_tests]
    statistics = [numpy.round(x[0], decimals) for x in ks_tests]

    ks_pvalues = dict(zip(numerical_columns, pvalues))
    ks_statistics = dict(zip(numerical_columns, statistics))

    # Cast numpy.nan and numpy.inf values (if any) to python None for JSON encoding
    ks_pvalues = fix_numpy_nans_and_infs_in_dict(
        values=ks_pvalues, test_name="Kolmogorov-Smirnov p-value"
    )
    if metric.lower() == "statistic":
        return fix_numpy_nans_and_infs_in_dict(
            values=ks_statistics, test_name="Kolmogorov-Smirnov statistic"
        )

    return ks_pvalues


def es_metric(
    df_1: pandas.DataFrame,
    df_2: pandas.DataFrame,
    numerical_columns: List[str],
    metric: Optional[str] = "p-value",
    decimals: Optional[int] = 4,
) -> dict:
    """
    Computes p_values corresponding to the ES (Epps-Singleton) two_sample statistic.

    Args:
        df_1 (pandas.DataFrame): baseline DataFrame.

        df_2 (pandas.DataFrame): sample DataFrame.

        numerical_columns (List[str]): list of numerical columns to compare.

        metric (str): Metric to return ('p-value' or 'statistic'). Default is 'p-value'.

        decimals (int): Number of decimals to round metrics to.

    Returns:
        Dictionary of p_values or statistics, one for each column in numerical_columns.
    """

    es_tests = []
    warnings.filterwarnings("error")
    warnings.filterwarnings("default", category=DeprecationWarning)

    for feat in numerical_columns:
        logger.info("Computing ES on numerical column %s", feat)
        x_samp = df_1.loc[:, feat]
        y_samp = df_2.loc[:, feat]

        # dropping nulls from feature in baseline
        if x_samp.isnull().values.any():
            logger.warning(
                "Numerical column %s in df_1 contains NULL values; these will be dropped to calculate es_metric.",
                feat,
            )
            x_samp = x_samp.dropna()

        # dropping nulls from feature in sample
        if y_samp.isnull().values.any():
            logger.warning(
                "Numerical column %s in df_2 contains NULL values; these will be dropped to calculate es_metric.",
                feat,
            )
            y_samp = y_samp.dropna()

        try:
            es_test = epps_singleton_2samp(x=x_samp, y=y_samp)

        except (numpy.linalg.LinAlgError, ValueError):
            # ValueError Possible culprits:
            # es_metric requires at least 5 values in both x_samp and y_samp

            logger.exception(
                "Unable to perform Epps-Singleton test on column %s. Setting ES metrics to None!",
                feat,
            )
            es_test = [None, None]

        except (UserWarning, RuntimeWarning, FloatingPointError):
            # UserWarning Possible culprits:
            # Estimated covariance matrix does not have full rank

            # RuntimeWarning, FloatingPointError Possible Culprits
            # Divide by zero encountered in true_divide (IQR=0)
            logger.exception(
                "Unable to perform Epps-Singleton test on column %s. Setting ES metrics to None!",
                feat,
            )
            es_test = [None, None]

        es_tests.append(es_test)

    pvalues = [x[1] for x in es_tests]
    # Rounding valid p-values
    for idx, _ in enumerate(pvalues):
        if pvalues[idx] is not None:
            pvalues[idx] = pvalues[idx].round(decimals)

    statistics = [x[0] for x in es_tests]
    # Rounding valid statistics
    for idx, _ in enumerate(statistics):
        if statistics[idx] is not None:
            statistics[idx] = statistics[idx].round(decimals)

    es_pvalues = dict(zip(numerical_columns, pvalues))
    es_statistics = dict(zip(numerical_columns, statistics))

    # Cast numpy.nan and numpy.inf values (if any) to python None for JSON encoding
    es_pvalues = fix_numpy_nans_and_infs_in_dict(
        values=es_pvalues, test_name="Epps-Singleton p-value"
    )
    if metric.lower() == "statistic":
        return fix_numpy_nans_and_infs_in_dict(
            values=es_statistics, test_name="Epps-Singleton statistic"
        )

    return es_pvalues


def js_metric(
    df_1: pandas.DataFrame,
    df_2: pandas.DataFrame,
    numerical_columns: List[str],
    categorical_columns: List[str],
    steps: Optional[int] = 100,
    decimals: Optional[int] = 4,
) -> dict:
    """
    Computes the Jensen-Shannon distances between columns of similar DataFrames.

    For categorical columns, the probability of each category will be
    computed separately for `df_1` and `df_2`, and the Jensen-Shannon distance
    between the 2 probability arrays will be computed.

    For numerical columns, the values will first be fitted into a Gaussian KDE
    separately for `df_1` and `df_2`, and a probability array
    will be sampled from them and compared with the Jensen-Shannon distance.

    Args:
        df_1 (pandas.DataFrame): baseline DataFrame.

        df_2 (pandas.DataFrame): sample DataFrame.

        numerical_columns (List[str]): list of numerical columns to compare.

        categorical_columns (List[str]): list of categorical columns.

        steps (int): Number of steps for numpy.linspace (numerical columns range).

        decimals (int): Number of decimals to round metrics to.

    Returns:
        Sorted list of tuples containing the column names and Jensen-Shannon distances.
    """

    warnings.filterwarnings("error")
    warnings.filterwarnings("default", category=DeprecationWarning)
    result = {}

    for col in categorical_columns:
        logger.info("Computing JS on categorical column %s", col)

        # to ensure similar order, concat before computing probability
        col_baseline = df_1[col].to_frame()
        col_sample = df_2[col].to_frame()
        col_baseline["source"] = "baseline"
        col_sample["source"] = "sample"

        # aggregate and convert to probability array
        arr = (
            pandas.concat([col_baseline, col_sample], ignore_index=True)
            .groupby([col, "source"])
            .size()
            .to_frame()
            .reset_index()
            .pivot(index=col, columns="source")
            .droplevel(0, axis=1)
        )
        arr_ = arr.div(arr.sum(axis=0), axis=1)
        arr_.fillna(0, inplace=True)

        # Calculate js distance
        result.update(
            {
                col: jensenshannon(
                    arr_["baseline"].to_numpy(), arr_["sample"].to_numpy()
                ).round(decimals)
            }
        )

    for col in numerical_columns:
        logger.info("Computing JS on numerical column %s", col)

        x_samp = df_1[col]
        y_samp = df_2[col]

        if x_samp.isna().values.any():
            logger.warning(
                "Numerical column %s in df_1 contains NULL values; these will be dropped to calculate js_metric.",
                col,
            )
            x_samp = x_samp.dropna()

        if y_samp.isna().values.any():
            logger.warning(
                "Numerical column %s in df_2 contains NULL values; these will be dropped to calculate js_metric.",
                col,
            )
            y_samp = y_samp.dropna()

        # Check to see if both columns are constants, and equal.
        # If so, set drift to zero without explict computation
        x_min, x_max, y_min, y_max = (
            x_samp.min(),
            x_samp.max(),
            y_samp.min(),
            y_samp.max(),
        )
        if (x_min == x_max) and (y_min == y_max) and (x_min == y_min):
            logger.info(
                "Values of column %s are constant and match in df_baseline and df_sample. Setting JS distance to 0 and skipping explicit computations.",
                col,
            )
            result.update({col: 0})
            continue

        # get range of values
        range_ = numpy.linspace(
            start=min(x_min, y_min),
            stop=max(x_max, y_max),
            num=steps,
        )
        try:
            # fit gaussian_kde then sample range
            arr_baseline = gaussian_kde(x_samp)(range_)
        except numpy.linalg.LinAlgError:
            logger.exception(
                "Unable to fit Gaussian KDE on column %s of df_baseline. Setting JS distance to None!",
                col,
            )
            result.update({col: None})
            continue

        try:
            arr_sample = gaussian_kde(y_samp)(range_)
        except numpy.linalg.LinAlgError:
            logger.exception(
                "Unable to fit Gaussian KDE on column %s of df_sample. Setting JS distance to None!",
                col,
            )
            result.update({col: None})
            continue

        arr_baseline = arr_baseline / numpy.sum(arr_baseline)
        arr_sample = arr_sample / numpy.sum(arr_sample)

        try:
            # calculate js distance
            result.update(
                {col: jensenshannon(arr_baseline, arr_sample).round(decimals)}
            )
        except RuntimeWarning:
            logger.exception(
                "Unable to compute JS distance on column %s. Setting JS distance to None! Invalid values in sqrt could indicate zero variance.",
                col,
            )
            result.update({col: None})

    # Cast numpy.nan and numpy.inf values (if any) to python None for JSON encoding
    result = fix_numpy_nans_and_infs_in_dict(
        values=result, test_name="Jensen-Shannon distance"
    )

    # using `or -1` to enable sorting with None values
    return dict(sorted(result.items(), key=lambda x: x[1] or -1, reverse=True))


def summarize_df(
    dataframe: pandas.DataFrame,
    numerical_columns: List[str],
    categorical_columns: List[str],
    decimals: Optional[int] = 4,
) -> dict:
    """
    Function to return a summary of a DataFrame (per column).

    Args:
        dataframe (pandas.DataFrame): input DataFrame.

        numerical_columns (List[str]): list of numerical columns to compare.

        categorical_columns (List[str]): list of categorical columns.

        decimals (int): Number of decimals to round metrics to.

    Returns:
        Summary of each column.
    """

    numerical_columns = list(numerical_columns) if numerical_columns else []
    categorical_columns = list(categorical_columns) if categorical_columns else []

    check_columns_in_dataframe(
        dataframe=dataframe, columns=numerical_columns + categorical_columns
    )

    numerical_df = dataframe.loc[:, numerical_columns]
    categorical_df = dataframe.loc[:, categorical_columns]

    # Summarize each of the DFs above
    if len(numerical_df.columns) > 0:
        numerical_df_summary = numerical_df.describe()
        numerical_nans = numerical_df.isna().sum()
        numerical_df_summary.loc["count_of_nulls"] = numerical_nans
    else:
        numerical_df_summary = pandas.DataFrame()

    if len(categorical_df.columns) > 0:
        # Force columns to be categorical in case where values are ints (0/1)
        categorical_df = categorical_df.astype("object")

        categorical_df_summary = categorical_df.describe()
        categorical_nans = categorical_df.isna().sum()
        categorical_df_summary.loc["count_of_nulls"] = categorical_nans
        # Change possible numpy ints to floats to make then JSON serializable (use .loc to avoid chained assignment)
        for col in categorical_df_summary.columns:
            for row in range(categorical_df_summary[col].shape[0]):
                val = categorical_df_summary[col].iloc[row]
                if isinstance(val, (numpy.int64, numpy.int32)):
                    idx = categorical_df_summary.index[row]
                    categorical_df_summary.loc[idx, col] = int(val)
    else:
        categorical_df_summary = pandas.DataFrame()

    return {
        "numerical_summary": numerical_df_summary.round(decimals),
        "categorical_summary": categorical_df_summary.round(decimals),
    }


def compare_dataframe_columns(
    df_1: pandas.DataFrame,
    df_2: pandas.DataFrame,
    numerical_columns: List[str],
    categorical_columns: List[str],
    decimals: Optional[int] = 4,
) -> dict:
    """
    Function to return a comparison of two DataFrames (per column).

    Args:
        df_1 (pandas.DataFrame): baseline DataFrame.

        df_2 (pandas.DataFrame): sample DataFrame.

        numerical_columns (List[str]): list of numerical columns to compare.

        categorical_columns (List[str]): list of categorical columns.

        decimals (int): Number of decimals to round metrics to.

    Returns:
        A dictionary of numerical and categorical comparisons between the columns \
        of the input DataFrames.
    """

    check_baseline_and_sample(
        df_baseline=df_1, df_sample=df_2, check_column_equality=False
    )
    check_columns_in_dataframe(
        dataframe=df_1, columns=numerical_columns + categorical_columns
    )

    df_1_summary = summarize_df(
        dataframe=df_1,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        decimals=decimals,
    )
    df_2_summary = summarize_df(
        dataframe=df_2,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        decimals=decimals,
    )

    numerical_comparisons = df_1_summary["numerical_summary"].compare(
        df_2_summary["numerical_summary"], keep_shape=True, keep_equal=True
    )
    numerical_comparisons.columns = numerical_comparisons.columns.set_levels(
        levels=["baseline", "sample"],
        level=1,
    )

    categorical_comparisons = df_1_summary["categorical_summary"].compare(
        df_2_summary["categorical_summary"], keep_shape=True, keep_equal=True
    )
    categorical_comparisons.columns = categorical_comparisons.columns.set_levels(
        levels=["baseline", "sample"],
        level=1,
    )

    numerical_comparisons_dict = dict(
        (col, {"baseline": None, "sample": None}) for col in numerical_columns
    )

    # Replace numpy.nan with None in metrics dictionaries (if present)
    for i, j in numerical_comparisons.columns:
        (numerical_comparisons_dict[i])[j] = fix_numpy_nans_and_infs_in_dict(
            numerical_comparisons[(i, j)].to_dict(), test_name="Summary"
        )

    categorical_comparisons_dict = dict(
        (col, {"baseline": None, "sample": None}) for col in categorical_columns
    )
    for i, j in categorical_comparisons.columns:
        (categorical_comparisons_dict[i])[j] = fix_numpy_nans_and_infs_in_dict(
            categorical_comparisons[(i, j)].to_dict(), test_name="Summary"
        )

    return {
        "numerical_comparisons": numerical_comparisons_dict,
        "categorical_comparisons": categorical_comparisons_dict,
    }


def kl_metric(
    df_1: pandas.DataFrame,
    df_2: pandas.DataFrame,
    numerical_columns: List[str],
    categorical_columns: List[str],
    num_buckets: int,
    decimals: Optional[int] = 4,
) -> dict:
    """
    Computes the Kullback-Leibler divergence between columns of similar DataFrames.

    Args:
        df_1 (pandas.DataFrame): Baseline DataFrame.

        df_2 (pandas.DataFrame): Sample DataFrame.

        numerical_columns (List[str]): List of numerical columns to compare.

        categorical_columns (List[str]): List of categorical columns to compare.

        num_buckets (int): Number of buckets used to bin values of numerical columns.

        decimals (int): Number of decimals to round metrics to.

    Returns:
        Dictionary of KL sum, one for each column.
    """

    result = {}
    for col in numerical_columns:
        logger.info("Computing KL on numerical column %s", col)

        # bucket column values by number of num_buckets,
        # then get the value counts of each bucket, then sort by index
        df_1_counts = (
            pandas.cut(df_1[col], num_buckets, include_lowest=True)
            .value_counts()
            .sort_index()
        )
        # convert values from counts to percentages to get distribution
        # of values by percentages of baseline dataframe
        b_dist = (df_1_counts / df_1.shape[0]).values

        # get the cut-offs from bucketing baseline
        b_index = [i.left for i in df_1_counts.index]

        # add right-most cut-off value from the index
        b_index.append(df_1_counts.index[-1].right)

        # get distribution of sample dataframe using baseline cut-offs
        s_dist = (
            pandas.cut(df_2[col], b_index, include_lowest=True)
            .value_counts()
            .sort_index()
            / df_2.shape[0]
        ).values

        # get element-wise KL values
        elemwise_kl = kl_div(b_dist, s_dist)

        # create element-wise dictionary of index and values
        # elemwise_dict = {k:v for k,v in zip(b_index, elemwise_kl)}
        # get sum of kl values
        kl_sum = numpy.sum(elemwise_kl)

        # if KL sum is infinity, reverse order of inputs
        if numpy.isinf(kl_sum):
            logger.warning(
                "KL sum for feature %s is Infinity. Recomputing with reversed inputs.",
                col,
            )
            elemwise_kl = kl_div(s_dist, b_dist)
            kl_sum = numpy.sum(elemwise_kl)

            if numpy.isinf(kl_sum):
                logger.warning(
                    "KL sum for feature %s is again Infinity with reversed inputs. Setting to None",
                    col,
                )
                kl_sum = None
            # Round to decimals if not None
            else:
                kl_sum = numpy.round(kl_sum, decimals)
        else:
            kl_sum = numpy.round(kl_sum, decimals)

        result.update({col: kl_sum})

    for col in categorical_columns:
        logger.info("Computing KL on categorical column %s", col)

        # bucket column of baseline dataset by all possible values
        df_1_counts = df_1[col].value_counts().sort_index()

        # get index of bucketed Series of baseline
        # in case there are some values that are unique to either baseline or sample datasets
        b_index = set(df_1_counts.index)

        # bucket column of sample dataset by all possible values
        df_2_counts = df_2[col].value_counts().sort_index()

        # get index of bucketed Series of sample
        s_index = set(df_2_counts.index)

        # get intersection of b_index and s_index (use list for pandas indexer)
        ind_intersection = list(b_index.intersection(s_index))

        # get distribution of baseline data and convert to percentages
        b_dist = (df_1_counts[ind_intersection] / df_1.shape[0]).values

        # get distribution of sample data and convert to percentages
        s_dist = (df_2_counts[ind_intersection] / df_2.shape[0]).values

        # get elemwise KL values
        elemwise_kl = kl_div(b_dist, s_dist)

        # create element-wise dictionary of index and values
        # elemwise_dict = {k:v for k,v in zip(b_index, elemwise_kl)}
        # get sum of kl values
        kl_sum = numpy.sum(elemwise_kl)

        # if KL sum is infinity, reverse order of inputs
        if numpy.isinf(kl_sum):
            logger.warning(
                "KL sum for feature %s is Infinity. Recomputing with reversed inputs.",
                col,
            )
            elemwise_kl = kl_div(s_dist, b_dist)
            kl_sum = numpy.sum(elemwise_kl)

            if numpy.isinf(kl_sum):
                logger.warning(
                    "KL sum for feature %s is again Infinity with reversed inputs. Setting to None",
                    col,
                )
                kl_sum = None
            # Round to decimals if not None
            else:
                kl_sum = numpy.round(kl_sum, decimals)
        else:
            kl_sum = numpy.round(kl_sum, decimals)

        result.update({col: kl_sum})

    # Cast numpy.nan and numpy.inf values (if any) to python None for JSON encoding
    result = fix_numpy_nans_and_infs_in_dict(
        values=result, test_name="Kullback-Leibler divergence"
    )

    return result


def ws_metric(
    df_1: pandas.DataFrame,
    df_2: pandas.DataFrame,
    numerical_columns: List[str],
    decimals: Optional[int] = 4,
) -> dict:
    """
    Computes the Wasserstein Distance between columns of similar DataFrames.

    Args:
        df_1 (pandas.DataFrame): Baseline DataFrame.

        df_2 (pandas.DataFrame): Sample DataFrame.

        numerical_columns (List[str]): List of numerical columns to compare.

        decimals (int): Number of decimals to round metrics to.

    Returns:
        Dictionary of Wasserstein distances, one for each column.

    """

    result = {}
    for col in numerical_columns:
        logger.info("Computing WS on numerical column %s", col)
        x_samp = df_1[col]
        y_samp = df_2[col]
        # check for null values in both dataframes for each column in numerical columns and drop null values
        if x_samp.isna().values.any():
            logger.warning(
                "Numerical column %s in df_1 contains NULL values; these will be dropped to calculate ws_metric.",
                col,
            )
            x_samp = x_samp.dropna()
        if y_samp.isna().values.any():
            logger.warning(
                "Numerical column %s in df_2 contains NULL values; these will be dropped to calculate ws_metric.",
                col,
            )
            y_samp = y_samp.dropna()

        # calculate the Wasserstein distance
        ws_dist = wasserstein_distance(u_values=x_samp, v_values=y_samp)

        # round to decimals, store distance into result dictionary
        result[col] = numpy.round(ws_dist, decimals)

    result = fix_numpy_nans_and_infs_in_dict(
        values=result, test_name="Wasserstein distance"
    )
    return result


def fix_numpy_nans_and_infs_in_dict(values: dict, test_name: str) -> dict:
    """A function to change all numpy.nan and numpy.inf values in a flat dictionary to python Nones.

    Args:
        values (dict): Input dict to fix.
        test_name (str):  Name of test that's calling this function.

    Returns:
        dict: Fixed dict.
    """

    for key, val in values.items():
        # If value is numeric (not None), check for numpy.nan and numpy.inf
        # If True, change to None, else keep unchanged
        if val is not None:
            try:  # Some values are not numeric
                if numpy.isnan(val):
                    values[key] = None
                elif numpy.isinf(val):
                    logger.warning(
                        "Infinity encountered while computing %s on column %s! Setting value to None.",
                        test_name,
                        key,
                    )
                    values[key] = None
            except TypeError:
                pass

    return values


if __name__ == "__main__":
    print(doctest.testmod())
    print()
