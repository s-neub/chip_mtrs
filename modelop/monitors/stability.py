"""This module provides a class to evaluate models on PSI and CSI metrics.

See `StabilityMonitor` for usage examples.

.. toctree::

.. autosummary::
   :toctree: _autosummary"""
import datetime
import doctest
import logging
import random
import string
from typing import List, Optional

# Third party packages
import numpy
import pandas
from statsmodels.stats.weightstats import DescrStatsW as dsw

import modelop.schema.infer as infer
from modelop.monitors.assertions import (
    check_baseline_and_sample,
    check_columns_in_dataframe,
    check_input_types, check_date_column,
)
from modelop.utils import get_min_max_values_keys_from_dict

pandas.options.mode.chained_assignment = "warn"
numpy.seterr(divide="raise")

logger = logging.getLogger(__name__)


class StabilityMonitor:
    """
    A class to evaluate the ML model on scored (containing predictions) baseline and sample datasets.

    Args:
        df_baseline (pandas.DataFrame): Pandas DataFrame of the baseline dataset.

        df_sample (pandas.DataFrame): Pandas DataFrame of the sample dataset.

        job_json (dict): JSON dictionary with the metadata of the model.

        predictors (List[str]): List of predictive features that are also drift candidates.

        feature_dataclass (dict): A dictionary of dataClass ('numerical' or 'categorical') keyed
            by feature names.

        special_values (dict): A dictionary of special values keyed by feature names. Optional.

        score_column (str): Column containing predicted values (as computed by
            underlying model). Optional. Default is None.

        weight_column (str): Column containing weight values. Optional. Default is None.

        date_column (str): Column containing dates for over time metrics.

        numerical_columns (List[str]): list of numerical columns to compare.

        decimals (int): Number of decimals to round metrics to. Default is 6.

    Raises:
        AssertionError: If input types are not met, or if `df_baseline` or `df_sample` do not
            match on column names and types, or if `score_column` is not in df_baseline.columns.

    Examples:
        Get Stability Analysis Table, Stability Index, Stability Chi-Square, and Stability KS
        metrics for numerical and categorical predictors, as well as the 'score' column.

        >>> import pandas
        >>> from pprint import pprint
        >>> from modelop.monitors.stability import StabilityMonitor

        >>> df_1 = pandas.DataFrame(
        ...     [
        ...         {'weight':1, 'amount':100, 'ownership':'own',  'score':0},
        ...         {'weight':1, 'amount':200, 'ownership':'rent', 'score':1},
        ...         {'weight':2, 'amount':150, 'ownership':'own',  'score':1},
        ...         {'weight':1, 'amount':125, 'ownership':'rent', 'score':0},
        ...         {'weight':3, 'amount':175, 'ownership':'own',  'score':1}
        ...     ]
        ... )

        >>> df_2 = pandas.DataFrame(
        ...     [
        ...         {'weight':2, 'amount':125, 'ownership':'rent', 'score':0},
        ...         {'weight':1, 'amount':150, 'ownership':'rent', 'score':1},
        ...         {'weight':1, 'amount':175, 'ownership':'own',  'score':0},
        ...         {'weight':3, 'amount':200, 'ownership':'own',  'score':0},
        ...         {'weight':1, 'amount':190, 'ownership':'rent', 'score':0}
        ...     ]
        ... )

        >>> stability_monitor = StabilityMonitor(
        ...         df_baseline=df_1,
        ...         df_sample=df_2,
        ...         predictors=['amount', 'ownership'],
        ...         feature_dataclass={
        ...             'amount':'numerical',
        ...             'ownership':'categorical',
        ...             'score':'categorical'
        ...         },
        ...         special_values={
        ...             'amount':[[100,125],[175]],
        ...             'ownership':[],
        ...             'score':[]
        ...         },
        ...         score_column='score',
        ...         weight_column='weight'
        ...     )

        >>> stability_metrics=stability_monitor.compute_stability_indices(
        ...     n_groups={
        ...         "amount": 2
        ...     },
        ...     group_cuts={
        ...         "score": [0,1]
        ...     },
        ...     flatten=True,
        ...     include_min_max_features=True,
        ... )

        >>> amount_analysis_table = pandas.DataFrame(
        ...    stability_metrics['stability'][0]['values']['amount']['stability_analysis_table']
        ... )

        >>> amount_analysis_table[
        ...     ['bucket', 'train_count', 'eval_count', 'ks_calc', 'chisq_calc', 'psi_calc']
        ... ]
                  bucket  train_count  eval_count  ks_calc  chisq_calc  psi_calc
        0       100 125             2           2    0.000    0.000000  0.000000
        1           175             3           1    0.250    0.166667  0.274653
        2  (-inf, 150.0]            2           1    0.375    0.062500  0.086643
        3   (150.0, inf]            1           4    0.000    1.125000  0.519860

        >>> stability_metrics['stability'][0]['values']['amount']['stability_index']
        0.881156

        >>> stability_metrics['stability'][0]['values']['amount']['stability_chisq']
        1.354167

        >>> stability_metrics['stability'][0]['values']['amount']['stability_ks']
        0.375

        >>> ownership_analysis_table = pandas.DataFrame(
        ...     stability_metrics['stability'][0]['values']['ownership']['stability_analysis_table']
        ... )

        >>> ownership_analysis_table[
        ...     ['bucket', 'train_count', 'eval_count', 'ks_calc', 'chisq_calc', 'psi_calc']
        ... ]
          bucket  train_count  eval_count  ks_calc  chisq_calc  psi_calc
        0    own            6           4     0.25    0.083333  0.101366
        1   rent            2           4     0.00    0.250000  0.173287

        >>> score_analysis_table = pandas.DataFrame(
        ...     stability_metrics['stability'][0]['values']['score']['stability_analysis_table']
        ... )

        >>> score_analysis_table[
        ...     ['bucket', 'train_count', 'eval_count', 'ks_calc', 'chisq_calc', 'psi_calc']
        ... ]
          bucket  train_count  eval_count  ks_calc  chisq_calc  psi_calc
        0      0            2           7    0.625    1.562500  0.782977
        1      1            6           1    0.000    0.520833  1.119850
        >>> stability_metrics["CSI_maxCSIValue"]
        0.881156
        >>> stability_metrics["CSI_maxCSIValueFeature"]
        'amount_CSI'
        >>> stability_metrics["CSI_minCSIValue"]
        0.274653
        >>> stability_metrics["CSI_minCSIValueFeature"]
        'ownership_CSI'


        The monitor can also be run on data that is missing a 'score' (model output) column.

        >>> import pandas
        >>> from pprint import pprint
        >>> from modelop.monitors.stability import StabilityMonitor

        >>> df_1 = pandas.DataFrame(
        ...     [
        ...         {'weight':1, 'amount':100, 'ownership':'own'},
        ...         {'weight':1, 'amount':200, 'ownership':'rent'},
        ...         {'weight':2, 'amount':150, 'ownership':'own'},
        ...         {'weight':1, 'amount':125, 'ownership':'rent'},
        ...         {'weight':3, 'amount':175, 'ownership':'own'}
        ...     ]
        ... )

        >>> df_2 = pandas.DataFrame(
        ...     [
        ...         {'weight':2, 'amount':125, 'ownership':'rent'},
        ...         {'weight':1, 'amount':150, 'ownership':'rent'},
        ...         {'weight':1, 'amount':175, 'ownership':'own'},
        ...         {'weight':3, 'amount':200, 'ownership':'own'},
        ...         {'weight':1, 'amount':190, 'ownership':'rent'}
        ...     ]
        ... )

        >>> stability_monitor = StabilityMonitor(
        ...         df_baseline=df_1,
        ...         df_sample=df_2,
        ...         predictors=['amount', 'ownership'],
        ...         feature_dataclass={
        ...             'amount':'numerical',
        ...             'ownership':'categorical',
        ...         },
        ...         special_values={
        ...             'amount':[[100,125],[175]],
        ...             'ownership':[],
        ...         },
        ...         score_column=None,
        ...         weight_column='weight'
        ...     )

        >>> stability_metrics=stability_monitor.compute_stability_indices(
        ...     n_groups={
        ...         "amount": 2
        ...     },
        ...     group_cuts={},
        ...     flatten=True,
        ...     include_min_max_features=True,
        ... )

        >>> amount_analysis_table = pandas.DataFrame(
        ...    stability_metrics['stability'][0]['values']['amount']['stability_analysis_table']
        ... )

        >>> amount_analysis_table[
        ...     ['bucket', 'train_count', 'eval_count', 'ks_calc', 'chisq_calc', 'psi_calc']
        ... ]
                  bucket  train_count  eval_count  ks_calc  chisq_calc  psi_calc
        0       100 125             2           2    0.000    0.000000  0.000000
        1           175             3           1    0.250    0.166667  0.274653
        2  (-inf, 150.0]            2           1    0.375    0.062500  0.086643
        3   (150.0, inf]            1           4    0.000    1.125000  0.519860

        >>> stability_metrics['stability'][0]['values']['amount']['stability_index']
        0.881156

        >>> stability_metrics['stability'][0]['values']['amount']['stability_chisq']
        1.354167

        >>> stability_metrics['stability'][0]['values']['amount']['stability_ks']
        0.375

        >>> ownership_analysis_table = pandas.DataFrame(
        ...     stability_metrics['stability'][0]['values']['ownership']['stability_analysis_table']
        ... )

        >>> ownership_analysis_table[
        ...     ['bucket', 'train_count', 'eval_count', 'ks_calc', 'chisq_calc', 'psi_calc']
        ... ]
          bucket  train_count  eval_count  ks_calc  chisq_calc  psi_calc
        0    own            6           4     0.25    0.083333  0.101366
        1   rent            2           4     0.00    0.250000  0.173287

        >>> stability_metrics["CSI_maxCSIValue"]
        0.881156
        >>> stability_metrics["CSI_maxCSIValueFeature"]
        'amount_CSI'
        >>> stability_metrics["CSI_minCSIValue"]
        0.274653
        >>> stability_metrics["CSI_minCSIValueFeature"]
        'ownership_CSI'
    """

    def __init__(
        self,
        df_baseline: pandas.DataFrame,
        df_sample: pandas.DataFrame,
        job_json: dict = None,
        predictors: List[str] = None,
        feature_dataclass: dict = None,
        special_values: dict = None,
        score_column: str = None,
        weight_column: str = None,
        date_column: str = None,
        numerical_columns: List[str] = None,
        decimals: Optional[int] = 6,
    ) -> None:

        check_baseline_and_sample(
            df_baseline=df_baseline, df_sample=df_sample, check_column_equality=False
        )

        if job_json is not None:
            logger.info(
                "Parameter 'job_json' is present and will be used to extract "
                "'predictors', 'feature_dataclass', 'special_values', "
                "'score_column', 'weight_column', 'numerical_columns' and 'date_column'."
            )
            input_schema_definition = infer.extract_input_schema(job_json)
            monitoring_parameters = infer.set_monitoring_parameters(
                schema_json=input_schema_definition, check_schema=True
            )
            predictors = monitoring_parameters["predictors"]
            feature_dataclass = monitoring_parameters["feature_dataclass"]
            special_values = monitoring_parameters["special_values"]
            score_column = monitoring_parameters["score_column"]
            weight_column = monitoring_parameters["weight_column"]
            numerical_columns = monitoring_parameters["numerical_columns"]
            date_column = monitoring_parameters["date_column"]
        else:
            logger.info(
                "Parameter 'job_json' it not present, attempting to use "
                "input features instead."
            )
            if (
                predictors is None
                or feature_dataclass is None
                or special_values is None
            ):
                missing_args_error = (
                    "Parameter 'job_json' is not present,"
                    " but one of 'predictors', 'feature_dataclass', 'special_values',"
                    " and 'numerical_columns' was not provided. "
                    "All of the above input parameters are"
                    " required if 'job_json' is not provided."
                )
                logger.error(missing_args_error)
                raise Exception(missing_args_error)

        assert isinstance(predictors, list), "predictors should be of type (list)."

        if numerical_columns is None:
            numerical_columns = []
            for k, v in feature_dataclass.items():
                if v == "numerical":
                    numerical_columns.append(k)

        assert isinstance(
            numerical_columns, list
        ), "numerical_columns should be of type (list)."

        check_input_types(
            inputs=[
                {"feature_dataclass": feature_dataclass},
                {"special_values": special_values},
            ],
            types=(dict),
        )

        if date_column is not None:
            check_input_types(inputs=[{"date_column": date_column}], types=(str))

        assert isinstance(
            score_column, (str, type(None))
        ), "score_column should be of type (str) or None."

        assert isinstance(
            weight_column, (str, type(None))
        ), "weight_column should be of type (str) or None."

        cols_to_scheck = predictors.copy()
        if score_column:
            self.score_provided = True
            cols_to_scheck.append(score_column)
        else:
            self.score_provided = False
            logger.info(
                "score_column was not provided and will be dropped from calculations."
            )
        if weight_column:
            cols_to_scheck.append(weight_column)

        if date_column is not None:
            cols_to_scheck.append(date_column)

        check_columns_in_dataframe(
            dataframe=df_baseline,
            columns=cols_to_scheck,
        )

        # Set default n_groups for each predictor and score
        n_groups = {}
        columns_to_bin = numerical_columns.copy()
        if self.score_provided:
            columns_to_bin.append(score_column)
        for feature in columns_to_bin:
            # if a feature has more than 1 unique value, set n_groups to 2; else set to 1
            feature_has_distinct_vlaues = int(
                (min(df_baseline[feature]) != max(df_baseline[feature]))
            )
            n_groups[feature] = 1 + feature_has_distinct_vlaues

        self.df_baseline = df_baseline
        self.df_sample = df_sample
        self.predictors = predictors
        self.feature_dataclass = feature_dataclass
        self.special_values = special_values
        self.score_column = score_column
        self.weight_column = weight_column
        self.n_groups = n_groups
        self.decimals = decimals
        self.date_column = date_column

        self.weight_var_name = "".join(
            random.choice(string.ascii_uppercase) for _ in range(6)
        )  # Create a weight variable name that won't clash with any feature or score name
        self.weighted_score_name = "".join(
            random.choice(string.ascii_uppercase) for _ in range(6)
        )  # Create a weighted score name that won't clash with any feature or score name

    def __str__(self):
        return self.__class__.__name__

    # Returns both Population Stability Index for a model score and
    # Characteristic Stability Indices for features.
    # Function requires that at least one feature name and the score column name be provided.

    def compute_stability_indices(
        self,
        n_groups: dict = None,
        group_cuts: dict = {},
        result_wrapper_key: str = "stability",
        flatten: bool = True,
        include_min_max_features: bool = True,
        include_over_time: bool = True,
    ):
        """
        Calculates Stability Analysis Table, Stability Index, Stability Chi-Square, and Stability
        KS metrics.

        Args:
            n_groups (dict): A dictionary keyed by feature names, indicating how many groups to
                feature values into. For example, n_groups={"feat_1": 2, "feat_2":5}.

            group_cuts (dict): A dictionary keyed by feature names, indicating bin cuts  for
                numerical features. For example, group_cuts={"feat_1": [0,1,5]}. If specified,
                a feature's group_cut takes precedence over a feature's n_group value.

            result_wrapper_key (str): Provides a key to wrap the result dict (flattened results are outside this entry).

            flatten (bool): Surfaces flat metrics to top level of return dictionary.

            include_min_max_features (bool): Provides the min and max values of the results,
                and their corresponding lags.

            include_over_time (bool): Includes a new section with metrics calculation over a prediction date column

        Returns:
            For each feature, a dictionary of the following tables and metrics:
                "stability_analysis_table", "stability_index", "stability_chisq"
                "stability_ks".
        """

        # If n_groups not provided in parameters, use n_groups on self
        if n_groups is None:
            n_groups = self.n_groups

        # Create DataFrames of train and eval with just the required variables,
        # namely score, features and weight, and get total number of cases for each

        train_calc_df_reference, total_train_cases = self.__process_weights(
            self, source="train", date_column=self.date_column
        )
        eval_calc_df_reference, total_eval_cases = self.__process_weights(
            self, source="eval", date_column=self.date_column
        )

        return_dict = {}
        stability_index_table = {}
        values_over_time = {}

        # Loop through feature names and model score name for the indices and calc tables
        stability_features = self.predictors.copy()
        if self.score_provided:
            stability_features.append(self.score_column)

        return_dict = self.calculate_stability_table(stability_features, train_calc_df_reference, eval_calc_df_reference, total_train_cases, total_eval_cases, group_cuts, n_groups)
        if include_over_time and self.date_column is not None:
            values_over_time = self.stability_over_time(self.calculate_stability_table, result_wrapper_key, stability_features, group_cuts, n_groups)

        stability_metrics = {
            "test_name": "Stability Analysis",
            "test_category": "stability",
            "test_type": "stability_analysis",
            "test_id": "stability_stability_analysis",
            "values": return_dict,
        }

        dict_of_CSI_features_values = {
            str(predictor + "_CSI"): stability_metrics["values"][predictor][
                "stability_index"
            ]
            for predictor in self.predictors
        }

        result = {}
        if include_min_max_features:

            min_max_dict = get_min_max_values_keys_from_dict(
                values_dict=dict_of_CSI_features_values
            )

            result["CSI_maxCSIValue"] = min_max_dict["max_value"]
            result["CSI_maxCSIValueFeature"] = min_max_dict["max_feature"]
            result["CSI_minCSIValue"] = min_max_dict["min_value"]
            result["CSI_minCSIValueFeature"] = min_max_dict["min_feature"]

        if include_over_time and self.date_column is not None:
            result.update(values_over_time)

        if flatten:
            if self.score_provided:
                result.update(
                    {
                        # Top-level metric
                        str(self.score_column + "_PSI"): stability_metrics["values"][
                            self.score_column
                        ]["stability_index"]
                    }
                )

            result.update(
                # Top-level metrics
                dict_of_CSI_features_values
            )

        result[result_wrapper_key] = [stability_metrics]

        return result

    def calculate_stability_table(self, stability_features, train_calc_df_reference, eval_calc_df_reference, total_train_cases, total_eval_cases, group_cuts, n_groups, index_only: bool = False):
        return_dict = {}
        stability_index_table = {}
        for fvarname in stability_features:
            # Set up empty DataFrames for calculations and output
            stability_index_table[fvarname] = pandas.DataFrame(
                {
                    "bucket": [],
                    "train_count": [],
                    "eval_count": [],
                    "train_percent": [],
                    "eval_percent": [],
                    "train_mean_score": [],
                    "eval_mean_score": [],
                    "score_mean_diff": [],
                    "train_cum_percent": [],
                    "eval_cum_percent": [],
                    "ks_calc": [],
                    "chisq_calc": [],
                    "psi_calc": [],
                }
            )

            # Reset train & eval DFs after processing each feature
            train_calc_df, eval_calc_df = (
                train_calc_df_reference,
                eval_calc_df_reference,
            )

            # Fill count, percent, mean_score of __MISSING__ bucket in stability_index_table
            # for train and eval DFs
            stability_index_table[fvarname].loc[0, "bucket"] = "__MISSING__"

            (
                stability_index_table,
                train_calc_df,
            ) = self.__fill_missing_counts_and_percentages(
                self,
                stability_index_table=stability_index_table,
                fvarname=fvarname,
                source="train",
                dataframe=train_calc_df,
                total_cases=total_train_cases,
            )

            (
                stability_index_table,
                eval_calc_df,
            ) = self.__fill_missing_counts_and_percentages(
                self,
                stability_index_table=stability_index_table,
                fvarname=fvarname,
                source="eval",
                dataframe=eval_calc_df,
                total_cases=total_eval_cases,
            )

            offset_rows = 1

            # Remove the row for __MISSING__ if either train or eval has no __MISSING__ values,
            # As drift comparison would be impossible
            train_nulls = stability_index_table[fvarname].loc[0, "train_count"]
            eval_nulls = stability_index_table[fvarname].loc[0, "eval_count"]

            # If neither DF has missing values, drop __MISSING__ bucket
            # An info-level message is enough
            if train_nulls == 0 and eval_nulls == 0:
                logger.info(
                    "Dropping the __MISSING__ bucket for %s: no NULLs in baseline or sample DFs.",
                    fvarname,
                )
                stability_index_table[fvarname].drop([0])
                offset_rows = 0

            # Deal with special values if present
            if self.special_values[fvarname] != []:
                for special_values_list in self.special_values[fvarname]:

                    stability_index_table[fvarname].loc[
                        offset_rows, "bucket"
                    ] = "".join([str(elem) + " " for elem in special_values_list])

                    (
                        stability_index_table,
                        train_calc_df,
                    ) = self.__adjust_for_special_values(
                        self,
                        stability_index_table=stability_index_table,
                        fvarname=fvarname,
                        source="train",
                        dataframe=train_calc_df,
                        special_values_list=special_values_list,
                        offset_rows=offset_rows,
                        total_cases=total_train_cases,
                    )

                    (
                        stability_index_table,
                        eval_calc_df,
                    ) = self.__adjust_for_special_values(
                        self,
                        stability_index_table=stability_index_table,
                        fvarname=fvarname,
                        source="eval",
                        dataframe=eval_calc_df,
                        special_values_list=special_values_list,
                        offset_rows=offset_rows,
                        total_cases=total_eval_cases,
                    )

                    offset_rows += 1

            # Get bin cuts from train data for numerical feature if group cuts not specified
            stability_index_table, variable_cuts = self.__adjust_for_cuts(
                self,
                stability_index_table=stability_index_table,
                fvarname=fvarname,
                source="train",
                dataframe=train_calc_df,
                train_calc_df=train_calc_df,
                total_cases=total_train_cases,
                offset_rows=offset_rows,
                group_cuts=group_cuts,
                n_groups=n_groups,
                variable_cuts=[],
            )

            stability_index_table, _ = self.__adjust_for_cuts(
                self,
                stability_index_table=stability_index_table,
                fvarname=fvarname,
                source="eval",
                dataframe=eval_calc_df,
                train_calc_df=train_calc_df,
                total_cases=total_eval_cases,
                offset_rows=offset_rows,
                group_cuts=group_cuts,
                n_groups=n_groups,
                variable_cuts=variable_cuts,
            )

            # Now do the calculations for KS, ChiSq and PSI
            return_dict = self.__compute_ks_psi_chisq(
                self,
                stability_index_table=stability_index_table,
                fvarname=fvarname,
                return_dict=return_dict,
                decimals=self.decimals,
            )
        return return_dict

    def stability_over_time(self, evaluation_function, result_wrapper_key, stability_features, group_cuts,
                            n_groups):
        sample_data = self.df_sample.copy()
        if self.date_column is not None and self.date_column in sample_data:
            sample_data = sample_data.set_index(pandas.to_datetime(check_date_column(sample_data, self.date_column)).dt.date)
            sample_data[self.date_column] = pandas.to_datetime(check_date_column(sample_data, self.date_column)).dt.date
            sample_data = sample_data.sort_index()

            unique_dates = sample_data[self.date_column].unique()

            data = {}
            for date in unique_dates:
                data_of_the_day = sample_data.loc[date]

                # Create DataFrames of train and eval with just the required variables,
                # namely score, features and weight, and get total number of cases for each

                train_calc_df_reference, total_train_cases = self.__process_weights(
                    self, source="train"
                )
                eval_calc_df_reference, total_eval_cases = self.__process_weights(
                    self, source="eval", date_column=self.date_column, date=date
                )
                dated_values = evaluation_function(stability_features, train_calc_df_reference, eval_calc_df_reference, total_train_cases, total_eval_cases, group_cuts, n_groups, True)

                str_date = str(date)
                for metric in dated_values:
                    # if metric in exclude_metrics:
                    #     continue
                    if metric not in data:
                        data[metric] = []
                    data[metric].append([str_date, dated_values[metric]["stability_index"]])

            over_time_key = result_wrapper_key + "_over_time"
            return {over_time_key: {
                "title": "Stability Over Time",
                "x_axis_label": "Day",
                "y_axis_label": "CSI",
                "data": data
            },
                "firstPredictionDate": str(unique_dates.min()),
                "lastPredictionDate": str(unique_dates.max()),
            }
        else:
            return {}

    @staticmethod
    def __process_weights(stability_monitor, source: str, date_column: str = None, date: datetime.date = None) -> tuple:
        """A function to process baseline and sample DataFrames to account for the weight column.

        Args:
            stability_monitor (modelop.monitors.drift.StabilityMonitor): Class instance.
            source (str): 'train' or 'eval'.

        Returns:
            tuple: Processed DataFrames (accounting for weights), and their record counts.
        """

        if source == "train":
            dataframe = stability_monitor.df_baseline.copy()
        else:  # when source == "eval":
            dataframe = stability_monitor.df_sample.copy()

        if date_column is not None:
            dataframe = dataframe.set_index(pandas.to_datetime(dataframe[date_column]).dt.date)
            dataframe[date_column] = pandas.to_datetime(dataframe[date_column]).dt.date
            dataframe = dataframe.sort_index()
            if date is not None:
                dataframe = dataframe.loc[[date]]
            dataframe = dataframe.reset_index(drop=True)

        if stability_monitor.weight_column is not None:
            dataframe[stability_monitor.weight_var_name] = dataframe.loc[
                :, stability_monitor.weight_column
            ]
        else:
            dataframe[stability_monitor.weight_var_name] = 1.0

        df_columns = stability_monitor.predictors + [stability_monitor.weight_var_name]
        if stability_monitor.score_provided:
            df_columns.append(stability_monitor.score_column)
        dataframe = dataframe.loc[:, df_columns]

        if stability_monitor.score_provided:
            dataframe[stability_monitor.weighted_score_name] = (
                dataframe[stability_monitor.weight_var_name]
                * dataframe[stability_monitor.score_column]
            )
        else:
            dataframe[stability_monitor.weighted_score_name] = None

        # Get total number of records (weighted) for calculating percentages
        total_cases = sum(dataframe[stability_monitor.weight_var_name])

        return (dataframe, total_cases)

    @staticmethod
    def __fill_missing_counts_and_percentages(
        stability_monitor,
        stability_index_table: dict,
        fvarname: str,
        source: str,
        dataframe: pandas.DataFrame,
        total_cases: int,
    ) -> tuple:
        """A function to fill stability_index_tables for train and eval counts.

        Args:
            stability_monitor (modelop.monitors.drift.StabilityMonitor): Class instance.
            stability_index_table (dict): Dictionary of all stability index tables (all fields).
            fvarname (str): Variable (field) under consideration.
            source (str): 'train' or 'eval'.
            dataframe (pandas.DataFrame): Train or Eval DataFrame.
            total_cases (int): Original length of `dataframe` before processing.

        Returns:
            tuple: Updated (`stability_index_table`, `dataframe`).
        """

        # Fill count of __MISSING__ bucket, taking into consideration weights column
        stability_index_table[fvarname].loc[0, source + "_count"] = sum(
            dataframe.loc[
                pandas.isna(dataframe[fvarname]), stability_monitor.weight_var_name
            ]
        )
        # Compute Percentage of __MISSING__ bucket
        stability_index_table[fvarname].loc[0, source + "_percent"] = (
            stability_index_table[fvarname].loc[0, source + "_count"] / total_cases
        )
        # Compute mean_score only if there are __MISSING__ values. Else set to numpy.nan
        if (stability_index_table[fvarname].loc[0, source + "_count"] > 0) and (
            stability_monitor.score_provided
        ):
            stability_index_table[fvarname].loc[0, source + "_mean_score"] = (
                sum(
                    dataframe.loc[
                        pandas.isna(dataframe[fvarname]),
                        stability_monitor.weighted_score_name,
                    ]
                )
                / stability_index_table[fvarname].loc[0, source + "_count"]
            )
        else:
            # stability_index_table[fvarname].loc[0, source + "_mean_score"] = numpy.nan
            stability_index_table[fvarname].loc[0, source + "_mean_score"] = None

        # Filter-out nulls
        dataframe = dataframe.loc[-pandas.isna(dataframe[fvarname]), :]

        return (stability_index_table, dataframe)

    @staticmethod
    def __adjust_for_special_values(
        stability_monitor,
        stability_index_table: dict,
        fvarname: str,
        source: str,
        dataframe: pandas.DataFrame,
        special_values_list: List,
        offset_rows: int,
        total_cases: int,
    ) -> tuple:
        """A function to update Stability Index Tables and Train/Eval DataFrames given special
        values for a particular field.

        Args:
            stability_monitor (modelop.monitors.drift.StabilityMonitor): Class instance.
            stability_index_table (dict): Dictionary of all stability index tables (all fields).
            fvarname (str): Variable (field) under consideration.
            source (str): 'train' or 'eval'.
            dataframe (pandas.DataFrame): Train or Eval DataFrame.
            special_values_list (List): List of special values for corresponding field (`fvarname`).
            offset_rows (int): Local variable indicating row position in `stability_index_table`.
            total_cases (int): Original length of `dataframe` before processing.

        Returns:
            tuple: Updated (`stability_index_table`, `dataframe`).
        """

        stability_index_table[fvarname].loc[offset_rows, source + "_count"] = sum(
            dataframe.loc[
                dataframe[fvarname].isin(special_values_list),
                stability_monitor.weight_var_name,
            ]
        )

        stability_index_table[fvarname].loc[offset_rows, source + "_percent"] = (
            stability_index_table[fvarname].loc[offset_rows, source + "_count"]
            / total_cases
        )

        if (
            stability_index_table[fvarname].loc[offset_rows, source + "_count"] > 0
        ) and (stability_monitor.score_provided):
            stability_index_table[fvarname].loc[offset_rows, source + "_mean_score"] = (
                sum(
                    dataframe.loc[
                        dataframe[fvarname].isin(special_values_list),
                        stability_monitor.weighted_score_name,
                    ]
                )
                / stability_index_table[fvarname].loc[offset_rows, source + "_count"]
            )
        else:
            stability_index_table[fvarname].loc[
                offset_rows, source + "_mean_score"
            ] = None

        dataframe = dataframe.loc[-dataframe[fvarname].isin(special_values_list), :]

        return (stability_index_table, dataframe)

    @staticmethod
    def __adjust_for_cuts(
        stability_monitor,
        stability_index_table: dict,
        fvarname: str,
        source: str,
        dataframe: pandas.DataFrame,
        train_calc_df: pandas.DataFrame,
        total_cases: int,
        offset_rows: int,
        group_cuts: dict,
        n_groups: dict,
        variable_cuts: Optional[List],
    ) -> dict:
        """A function to update Stability Index Tables given `group_cuts` and/or `n_groups`
        for a particular field.

        Args:
            stability_monitor (modelop.monitors.drift.StabilityMonitor): Class instance.
            stability_index_table (dict): Dictionary of all stability index tables (all fields).
            fvarname (str): Variable (field) under consideration.
            source (str): 'train' or 'eval'.
            dataframe (pandas.DataFrame): Train or Eval DataFrame.
            train_calc_df (pandas.DataFrame): Train DataFrame.
            total_cases (int): Original length of `dataframe` before processing.
            offset_rows (int): Local variable indicating row position in `stability_index_table`.
            group_cuts (dict): Argument to `compute_stability_indices` method.
            n_groups (dict): Argument to `compute_stability_indices` method.
            variable_cuts (List): variable cuts for numerical features

        Returns:
            dict: Updated `stability_index_table`.
        """

        dataframe = dataframe.copy(deep=True)

        if stability_monitor.feature_dataclass[fvarname] == "numerical":

            # Compute variable_cuts only when DF is train/baseline dataset
            if source == "train":
                if fvarname in group_cuts.keys():
                    variable_cuts = group_cuts[fvarname]
                    variable_cuts = [-numpy.inf] + variable_cuts + [numpy.inf]
                else:
                    variable_cuts = list(
                        dsw(
                            train_calc_df[fvarname],
                            train_calc_df[stability_monitor.weight_var_name],
                        ).quantile(probs=numpy.linspace(0, 1, n_groups[fvarname] + 1))
                    )
                    variable_cuts[0] = -numpy.inf
                    variable_cuts[len(variable_cuts) - 1] = numpy.inf

            # Fill stability_index_table[fvarname] according to variable_cuts
            # if source is eval, variable cuts are passed as argument (computed on train)
            dataframe.loc[:, "Var_Range"] = pandas.cut(
                dataframe.loc[:, fvarname],
                bins=variable_cuts,
                duplicates="drop",
                include_lowest=True,
            )
            # aggregate by Sum
            bivar_tbl = dataframe.groupby(["Var_Range"]).sum()

        # If feature is categorical, use each value as a bucket
        else:
            bivar_tbl = dataframe.groupby([fvarname]).sum()
            bivar_tbl.index = pandas.CategoricalIndex(bivar_tbl.index)

        bivar_tbl["Var_Range"] = bivar_tbl.index

        # populate stability_index_table given the baseline dataset
        if source == "train":
            for i in range(bivar_tbl.shape[0]):
                stability_index_table[fvarname].loc[
                    i + offset_rows, "bucket"
                ] = bivar_tbl.loc[i, "Var_Range"]

                stability_index_table[fvarname].loc[
                    i + offset_rows, "train_count"
                ] = bivar_tbl.loc[i, stability_monitor.weight_var_name]

                stability_index_table[fvarname].loc[
                    i + offset_rows, "train_percent"
                ] = (bivar_tbl.loc[i, stability_monitor.weight_var_name] / total_cases)

                # TODO: Consider edge case where denominator=0 (no records in bucket)
                if stability_monitor.score_provided:
                    stability_index_table[fvarname].loc[
                        i + offset_rows, "train_mean_score"
                    ] = (
                        bivar_tbl.loc[i, stability_monitor.weighted_score_name]
                        / stability_index_table[fvarname].loc[
                            i + offset_rows, "train_count"
                        ]
                    )
                else:
                    stability_index_table[fvarname].loc[
                        i + offset_rows, "train_mean_score"
                    ] = None

                stability_index_table[fvarname].loc[i + offset_rows, "eval_count"] = 0
                stability_index_table[fvarname].loc[i + offset_rows, "eval_percent"] = 0
                stability_index_table[fvarname].loc[
                    i + offset_rows, "eval_mean_score"
                ] = None

        # If the dataset is the sample/eval dataset, do:
        else:

            for i in range(bivar_tbl.shape[0]):
                # Current feature value in eval dataset
                fvarname_value = bivar_tbl.loc[i, "Var_Range"]

                # value in eval has been encountered in train
                if fvarname_value in stability_index_table[fvarname]["bucket"].values:

                    index = stability_index_table[fvarname].index
                    condition = (
                        stability_index_table[fvarname]["bucket"] == fvarname_value
                    )
                    row_idx = index[condition].to_list()[0]

                    stability_index_table[fvarname].loc[
                        row_idx, "eval_count"
                    ] = bivar_tbl.loc[i, stability_monitor.weight_var_name]

                    stability_index_table[fvarname].loc[row_idx, "eval_percent"] = (
                        bivar_tbl.loc[i, stability_monitor.weight_var_name]
                        / total_cases
                    )

                    if stability_monitor.score_provided and stability_index_table[fvarname].loc[row_idx, "eval_count"] > 0:
                        stability_index_table[fvarname].loc[
                            row_idx, "eval_mean_score"
                        ] = (
                            bivar_tbl.loc[i, stability_monitor.weighted_score_name]
                            / stability_index_table[fvarname].loc[row_idx, "eval_count"]
                        )
                    else:
                        stability_index_table[fvarname].loc[
                            row_idx, "eval_mean_score"
                        ] = None

                # new value encountered
                else:
                    row_idx = stability_index_table[fvarname].shape[0]
                    stability_index_table[fvarname].loc[
                        row_idx, "bucket"
                    ] = fvarname_value

                    # set values for train
                    stability_index_table[fvarname].loc[row_idx, "train_count"] = 0
                    stability_index_table[fvarname].loc[row_idx, "train_percent"] = 0
                    # stability_index_table[fvarname].loc[row_idx, "train_mean_score"] = numpy.nan
                    stability_index_table[fvarname].loc[
                        row_idx, "train_mean_score"
                    ] = None

                    # fill values for eval
                    stability_index_table[fvarname].loc[
                        row_idx, "eval_count"
                    ] = bivar_tbl.loc[i, stability_monitor.weight_var_name]

                    stability_index_table[fvarname].loc[row_idx, "eval_percent"] = (
                        bivar_tbl.loc[i, stability_monitor.weight_var_name]
                        / total_cases
                    )

                    if stability_monitor.score_provided and stability_index_table[fvarname].loc[row_idx, "eval_count"] > 0:
                        stability_index_table[fvarname].loc[
                            row_idx, "eval_mean_score"
                        ] = (
                            bivar_tbl.loc[i, stability_monitor.weighted_score_name]
                            / stability_index_table[fvarname].loc[row_idx, "eval_count"]
                        )
                    else:
                        stability_index_table[fvarname].loc[
                            row_idx, "eval_mean_score"
                        ] = None

        return stability_index_table, variable_cuts

    @staticmethod
    def __compute_ks_psi_chisq(
        stability_monitor,
        stability_index_table: dict,
        fvarname: str,
        return_dict: dict,
        decimals: Optional[int] = 6,
    ) -> dict:
        """A function to compute KS, Chi_Square, and PSI for a given field, and then update
        the overall output of the `compute_stability_indices` method.

        Args:
            stability_monitor (modelop.monitors.drift.StabilityMonitor): Class instance.
            fvarname (str): Variable (field) under consideration.
            return_dict (dict): Dictionary of computed metrics, indexed by field (`fvarname`).
            decimals (int): Number of decimals to round metrics to. Default is 6.

        Returns:
            dict: Updated `return_dict`.
        """

        # Now do the calculations for KS, ChiSq and PSI
        for i in range(stability_index_table[fvarname].shape[0]):
            if i == 0:
                stability_index_table[fvarname].loc[
                    i, "train_cum_percent"
                ] = stability_index_table[fvarname].loc[i, "train_percent"]

                stability_index_table[fvarname].loc[
                    i, "eval_cum_percent"
                ] = stability_index_table[fvarname].loc[i, "eval_percent"]

            else:
                stability_index_table[fvarname].loc[i, "train_cum_percent"] = (
                    stability_index_table[fvarname].loc[i - 1, "train_cum_percent"]
                    + stability_index_table[fvarname].loc[i, "train_percent"]
                )

                stability_index_table[fvarname].loc[i, "eval_cum_percent"] = (
                    stability_index_table[fvarname].loc[i - 1, "eval_cum_percent"]
                    + stability_index_table[fvarname].loc[i, "eval_percent"]
                )

            if stability_monitor.score_provided:
                stability_index_table[fvarname].loc[i, "score_mean_diff"] = (
                    stability_index_table[fvarname].loc[i, "eval_mean_score"]
                    - stability_index_table[fvarname].loc[i, "train_mean_score"]
                )
            else:
                stability_index_table[fvarname].loc[i, "score_mean_diff"] = None

            stability_index_table[fvarname].loc[i, "ks_calc"] = abs(
                stability_index_table[fvarname].loc[i, "train_cum_percent"]
                - stability_index_table[fvarname].loc[i, "eval_cum_percent"]
            )

            if stability_index_table[fvarname].loc[i, "train_percent"] > 0:
                stability_index_table[fvarname].loc[i, "chisq_calc"] = (
                    numpy.power(
                        stability_index_table[fvarname].loc[i, "train_percent"]
                        - stability_index_table[fvarname].loc[i, "eval_percent"],
                        2,
                    )
                    / stability_index_table[fvarname].loc[i, "train_percent"]
                )

                if stability_index_table[fvarname].loc[i, "eval_percent"] == 0:
                    logger.warning(
                        "Zero Encountered in log(eval_percent/train_percent) for %s=%s. \
                        \ntrain_percent=%f \neval_percent=%f. \
                        \nSetting psi_calc to None.\
                        This bucket will be IGNORED when aggregating PSI!",
                        fvarname,
                        stability_index_table[fvarname].loc[i, "bucket"],
                        stability_index_table[fvarname].loc[i, "train_percent"],
                        stability_index_table[fvarname].loc[i, "eval_percent"],
                    )
                    stability_index_table[fvarname].loc[i, "psi_calc"] = None

                else:
                    stability_index_table[fvarname].loc[i, "psi_calc"] = (
                        stability_index_table[fvarname].loc[i, "eval_percent"]
                        - stability_index_table[fvarname].loc[i, "train_percent"]
                    ) * numpy.log(
                        stability_index_table[fvarname].loc[i, "eval_percent"]
                        / (stability_index_table[fvarname].loc[i, "train_percent"])
                    )

            else:
                logger.warning(
                    "For %s=%s, \
                    \ntrain_percent=0. \
                    \nSetting psi_calc and chisq_calc to None (Unable to Divide).\
                    This bucket will be IGNORED when aggregating PSI and CHI_SQUARE!",
                    fvarname,
                    stability_index_table[fvarname].loc[i, "bucket"],
                )

                stability_index_table[fvarname].loc[i, "chisq_calc"] = None
                stability_index_table[fvarname].loc[i, "psi_calc"] = None

        # Cast buckets as strings - Needed especially in converting numpy intervals to strings
        stability_index_table[fvarname]["bucket"] = stability_index_table[fvarname][
            "bucket"
        ].astype(str)

        # Round float columns to decimals
        stability_index_table[fvarname] = stability_index_table[fvarname].round(
            decimals=decimals
        )

        # Cast count columns as ints
        stability_index_table[fvarname][
            ["train_count", "eval_count"]
        ] = stability_index_table[fvarname][["train_count", "eval_count"]].astype(int)

        # Add metrics to return_dict
        return_dict[fvarname] = {
            "stability_analysis_table": fix_numpy_nans_in_dict_array(
                stability_index_table[fvarname].to_dict(orient="records")
            ),
            "stability_index": round(
                stability_index_table[fvarname]["psi_calc"].sum(), decimals
            )
            if not numpy.isnan(stability_index_table[fvarname]["psi_calc"].sum())
            else None,
            "stability_chisq": round(
                stability_index_table[fvarname]["chisq_calc"].sum(), decimals
            )
            if not numpy.isnan(stability_index_table[fvarname]["chisq_calc"].sum())
            else None,
            "stability_ks": round(
                stability_index_table[fvarname]["ks_calc"].max(), decimals
            )
            if not numpy.isnan(stability_index_table[fvarname]["ks_calc"].max())
            else None,
        }

        return return_dict


def fix_numpy_nans_in_dict_array(values: List[dict]) -> List[dict]:
    """A function to iterate over dictionaries in an array,
    and change all numpy.nan values to a python None.

    Args:
        values (List[dict]): Input list of dicts to fix.

    Returns:
        List[dict]: Fixed list.
    """

    # This will hold return list
    fixed_values = []
    # iterate over dicts in list
    for group in values:
        fixed_group = {}
        # Iterate over items in dict
        for key, val in group.items():
            # Some values are strings, skip over them (no need to fix)
            # If value is numeric, check for numpy.nan;
            # If True, change to None, else keep unchanged
            if key == "bucket":
                fixed_group[key] = val
            else:
                fixed_group[key] = val if not numpy.isnan(val) else None

        # Dictionary is now fixed. Add to return list
        fixed_values.append(fixed_group)

    return fixed_values


if __name__ == "__main__":
    print(doctest.testmod())
    print()
