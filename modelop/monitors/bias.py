"""This module provides several monitors for ethical bias/fairness using Aequitas.

See `BiasMonitor` for usage examples.

.. toctree::

.. autosummary::
   :toctree: _autosummary
"""
from copy import deepcopy
import doctest
import json
import logging
from typing import List, Optional, Union

# Third party packages
import numpy
import pandas
from aequitas.bias import Bias
from aequitas.group import Group
from aequitas.preprocessing import preprocess_input_df

import modelop.schema.infer as infer
from modelop.monitors.assertions import (
    check_columns_in_dataframe,
    check_date_column,
    check_input_types,
    check_pre_defined_metric,
)
from modelop.utils import fix_numpy_nans_in_dict_array

pandas.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


class BiasMonitor:
    """
    Evaluates a ML classification model's bias/group metrics on a scored & labeled dataset.

    Args:
        dataframe (pandas.DataFrame): Pandas DataFrame of scored and labeled data.

        job_json (dict): JSON dictionary with the metadata of the model.

        score_column (str): Column containing predicted values (as computed by underlying model).

        label_column (str): Column containing actual values (ground truths).

        protected_classes (Optional[List[dict]]): List of dictionaries, each defining a protected class,
            of the form {"protected_class": (str) , "reference_group": (str), "numerical_cutoffs": (List)}.
            "reference_group" is one of the values of protected_class, used for reference. See examples below.
            "numerical_cutoffs" is a list of float cutoffs for numerical columns to be bucketed into. See examples below.

        decimals (int): Number of decimals to round metrics to. Default is 4.

        date_column (str): Column containing dates for over time metrics.

    Raises:
        AssertionError: If `dataframe` is not a pandas DataFrame, or if the other parameters are not
            strings, or if `score_column`, `label_column`, or protected classes are not in
            dataframe.columns.

    Examples:
        Get group metrics on a labeled and scored dataset with protected_class='gender':

        >>> import pandas
        >>> from pprint import pprint
        >>> from modelop.monitors.bias import BiasMonitor

        >>> dataframe = pandas.DataFrame(
        ...     [
        ...         {'gender':'male', 'score':0, 'label':0},
        ...         {'gender':'male', 'score':1, 'label':1},
        ...         {'gender':'male', 'score':0, 'label':0},
        ...         {'gender':'male', 'score':1, 'label':0},
        ...         {'gender':'female', 'score':0, 'label':0},
        ...         {'gender':'female', 'score':1, 'label':1},
        ...         {'gender':'female', 'score':0, 'label':1},
        ...         {'gender':'female', 'score':1, 'label':1}
        ...     ]
        ... )

        >>> bias_monitor = BiasMonitor(
        ...     dataframe=dataframe,
        ...     score_column='score',
        ...     label_column='label',
        ...     protected_classes=[{"protected_class": "gender"}],
        ... )

        >>> pprint(
        ...     bias_monitor.compute_group_metrics(
        ...         pre_defined_test='aequitas_group',
        ...     ),
        ...     sort_dicts=False
        ... )
        {'bias': [{'test_name': 'Aequitas Group',
                   'test_category': 'bias',
                   'test_type': 'group',
                   'protected_class': 'gender',
                   'test_id': 'bias_group_gender',
                   'reference_group': None,
                   'values': [{'attribute_name': 'gender',
                               'attribute_value': 'female',
                               'tpr': 0.6667,
                               'tnr': 1.0,
                               'for': 0.5,
                               'fdr': 0.0,
                               'fpr': 0.0,
                               'fnr': 0.3333,
                               'npv': 0.5,
                               'precision': 1.0,
                               'ppr': 0.5,
                               'pprev': 0.5,
                               'prev': 0.75},
                              {'attribute_name': 'gender',
                               'attribute_value': 'male',
                               'tpr': 1.0,
                               'tnr': 0.6667,
                               'for': 0.0,
                               'fdr': 0.5,
                               'fpr': 0.3333,
                               'fnr': 0.0,
                               'npv': 1.0,
                               'precision': 0.5,
                               'ppr': 0.5,
                               'pprev': 0.5,
                               'prev': 0.25}]}]}

        Get bias metrics on the same dataset with protected_class='gender' and reference_group='male':

        >>> pprint(
        ...     bias_monitor.compute_bias_metrics(
        ...         pre_defined_test='aequitas_bias',
        ...         flatten=False,
        ...         include_min_max_features=False
        ...     ),
        ...     sort_dicts=False
        ... )
        get_disparity_predefined_group()
        {'bias': [{'test_name': 'Aequitas Bias',
                   'test_category': 'bias',
                   'test_type': 'bias',
                   'protected_class': 'gender',
                   'test_id': 'bias_bias_gender',
                   'reference_group': 'male',
                   'thresholds': None,
                   'values': [{'attribute_name': 'gender',
                               'attribute_value': 'female',
                               'ppr_disparity': 1.0,
                               'pprev_disparity': 1.0,
                               'precision_disparity': 2.0,
                               'fdr_disparity': 0.0,
                               'for_disparity': 10.0,
                               'fpr_disparity': 0.0,
                               'fnr_disparity': 10.0,
                               'tpr_disparity': 0.6667,
                               'tnr_disparity': 1.5,
                               'npv_disparity': 0.5},
                              {'attribute_name': 'gender',
                               'attribute_value': 'male',
                               'ppr_disparity': 1.0,
                               'pprev_disparity': 1.0,
                               'precision_disparity': 1.0,
                               'fdr_disparity': 1.0,
                               'for_disparity': None,
                               'fpr_disparity': 1.0,
                               'fnr_disparity': None,
                               'tpr_disparity': 1.0,
                               'tnr_disparity': 1.0,
                               'npv_disparity': 1.0}]}]}

        Compute group metrics on protected_class='age', with numerical values, and numerical cutoffs:

        >>> dataframe = pandas.DataFrame(
        ...     [
        ...         {"gender": "male", "age": 10, "prediction": 0, "label": 0},
        ...         {"gender": "male", "age": 18, "prediction": 0, "label": 0},
        ...         {"gender": "female", "age": 20, "prediction": 0, "label": 0},
        ...         {"gender": "female", "age": 25, "prediction": 0, "label": 0},
        ...         {"gender": "male", "age": 30, "prediction": 1, "label": 1},
        ...         {"gender": "female", "age": 40, "prediction": 1, "label": 1},
        ...         {"gender": "female", "age": 42, "prediction": 1, "label": 1},
        ...         {"gender": "male", "age": 50, "prediction": 0, "label": 1},
        ...         {"gender": "male", "age": 55, "prediction": 0, "label": 1},
        ...         {"gender": "female", "age": 60, "prediction": 0, "label": 1},
        ...         {"gender": "male", "age": 70, "prediction": 1, "label": 0},
        ...         {"gender": "female", "age": 80, "prediction": 1, "label": 1},
        ...     ]
        ... )

        >>> # Instantiate Bias Monitor
        >>> numerical_bias_monitor = BiasMonitor(
        ...     dataframe=dataframe,
        ...     score_column="prediction",
        ...     label_column="label",
        ...     protected_classes=[{"protected_class":"age", "numerical_cutoffs": [40]}],
        ... )

        >>> pprint(
        ...     numerical_bias_monitor.compute_bias_metrics(
        ...         pre_defined_test='aequitas_bias',
        ...         flatten=False,
        ...         include_min_max_features=False
        ...     ),
        ...     sort_dicts=False
        ... )
        get_disparity_predefined_group()
        {'bias': [{'test_name': 'Aequitas Bias',
                   'test_category': 'bias',
                   'test_type': 'bias',
                   'protected_class': 'age_bucketed',
                   'test_id': 'bias_bias_age_bucketed',
                   'reference_group': '(-inf, 40)',
                   'thresholds': None,
                   'values': [{'attribute_name': 'age_bucketed',
                               'attribute_value': '(-inf, 40)',
                               'ppr_disparity': 1.0,
                               'pprev_disparity': 1.0,
                               'precision_disparity': 1.0,
                               'fdr_disparity': None,
                               'for_disparity': None,
                               'fpr_disparity': None,
                               'fnr_disparity': None,
                               'tpr_disparity': 1.0,
                               'tnr_disparity': 1.0,
                               'npv_disparity': 1.0},
                              {'attribute_name': 'age_bucketed',
                               'attribute_value': '[40, +inf)',
                               'ppr_disparity': 4.0,
                               'pprev_disparity': 2.8571,
                               'precision_disparity': 0.75,
                               'fdr_disparity': 10.0,
                               'for_disparity': 10.0,
                               'fpr_disparity': 10.0,
                               'fnr_disparity': 10.0,
                               'tpr_disparity': 0.5,
                               'tnr_disparity': 0.0,
                               'npv_disparity': 0.0}]}]}
    """

    def __init__(
        self,
        dataframe: pandas.DataFrame,
        job_json: Optional[dict] = None,
        score_column: Optional[str] = None,
        label_column: Optional[str] = None,
        protected_classes: Optional[List[dict]] = None,
        positive_class_label: Optional[Union[int, bool, str]] = None,
        decimals: int = 4,
        date_column: str = None
    ) -> None:

        assert isinstance(
            dataframe, pandas.DataFrame
        ), "dataframe should be of type (pandas.DataFrame)."

        if job_json is not None:
            logger.info(
                "Parameter 'job_json' is present and will be used to extract "
                "'score_column', 'label_column', and 'protected_class'."
            )
            input_schema_definition = infer.extract_input_schema(job_json)
            monitoring_parameters = infer.set_monitoring_parameters(
                schema_json=input_schema_definition, check_schema=True
            )
            score_column = monitoring_parameters["score_column"]
            label_column = monitoring_parameters["label_column"]
            date_column = monitoring_parameters["date_column"]
            # Retrieving Job Parameters
            self.reference_groups = {}
            try:
                extracted_params = json.loads(job_json['rawJson'])['jobParameters']
                if not ("reference_groups" in extracted_params and isinstance(extracted_params, dict)):
                    logger.info("Reference groups not detected, will default to unique values "
                                "per protected class.")
                else:
                    self.reference_groups = extracted_params["reference_groups"]
                    if not isinstance(self.reference_groups, dict):
                        logger.error("reference_groups job parameter was not a dictionary. Please check json structure of job parameters.")
                        raise TypeError("reference_groups was not expected type of dict/dictionary")
                    logger.info("Reference groups detected, will try to use per protected class.")
            except Exception as err:
                self.reference_groups = {}
                logger.warning(f"Was unable to retrieve reference_groups from job parameters due to {err}.\nMoving to default behavior.")
            protected_classes = []
            for feature in monitoring_parameters["protected_classes"]:
                protected_classes.append(
                    {
                        "protected_class": feature,
                        "reference_group": self.reference_groups.get(feature, None),
                        "numerical_cutoffs": None,
                    }
                )

        else:
            if (
                score_column is None
                or label_column is None
                or protected_classes is None
            ):
                missing_args_error = (
                    "Parameter 'job_json' is not present, "
                    "but one of 'score_column', 'label_column', or 'protected_classes' was not provided. "
                    "'score_column', 'label_column', and 'protected_classes' input parameters are "
                    "required if 'job_json' is not provided."
                )
                logger.error(missing_args_error)
                raise Exception(missing_args_error)

        # Check for protected classes, in case job_json does not have them
        if protected_classes == []:
            raise ValueError("Input Schema contains no Protected Classes!")

        check_input_types(
            inputs=[
                {"score_column": score_column},
                {"label_column": label_column},
            ],
            types=(str),
        )

        check_input_types(
            inputs=[
                {"protected_classes": protected_classes},
            ],
            types=(List, type(None)),
        )

        if date_column is not None:
            check_columns_in_dataframe(
                dataframe=dataframe, columns=[score_column, label_column, date_column]
            )
        else:
            check_columns_in_dataframe(
                dataframe=dataframe, columns=[score_column, label_column]
            )
        if date_column is not None:
            check_input_types(inputs=[{"date_column": date_column}], types=(str))

        # Check for and drop nulls
        dataframe = self.__check_and_drop_nulls(
            dataframe, protected_classes, score_column, label_column
        )

        for idx, p_class in enumerate(protected_classes):
            feature = p_class["protected_class"]
            # Check to see if each protected class in list of protected classes is in dataframe
            check_columns_in_dataframe(dataframe=dataframe, columns=[feature])

            # If numerical_buckets are provided, bucket protected class
            n_cutoff = p_class.get("numerical_cutoffs", None)
            if n_cutoff is not None:
                # Check if numerical_cutoffs is an empty list
                if n_cutoff == []:
                    raise ValueError(
                        "'numerical_cutoffs' is an empty list! Acceptable options are `None` or a non-empty list of numerical values."
                    )

                # Create cutoffs for buckets using the provided numerical_buckets list
                cutoffs_for_buckets = self.__make_cutoff_dictionary(n_cutoff)

                # Create new column that holds bucketed values
                # Opt to using .format with %s syntax seems to not give correct results
                dataframe.loc[:, f"{feature}_bucketed"] = self.__label_cutoffs(
                    dataframe.loc[:, feature], cutoffs_for_buckets
                )
                # Reassign self.protected_class to the bucketed column
                protected_classes[idx]["protected_class"] = f"{feature}_bucketed"

                feature = feature + "_bucketed"

        if positive_class_label is None:
            # Get class labels
            class_labels = list(dataframe[score_column].unique())
            if len(class_labels) == 0:
                raise KeyError("Dataframe cannot be empty!")
            elif len(class_labels) > 2:
                raise ValueError(
                    "Expected no more than 2 labels for a binary classification model."
                )
            else:
                if set(class_labels) <= {0, 1}:
                    positive_class_label = 1
                    logger.warning(
                        "positive_class_label was not provided. Setting it to 1.",
                    )
                elif set(class_labels) <= {True, False}:
                    positive_class_label = True
                    logger.warning(
                        "positive_class_label was not provided. Setting it to True.",
                    )
                else:
                    positive_class_label = dataframe.loc[0, score_column]
                    logger.warning(
                        "positive_class_label was not provided. Setting it to the first observed value in score_column %s",
                        positive_class_label,
                    )

        columns_to_keep = [
            class_dict["protected_class"] for class_dict in protected_classes
        ] + [score_column, label_column] + ([date_column] if date_column is not None else [])
        logger.info("Restricting dataframe to columns %s", columns_to_keep)

        self.dataframe = dataframe[columns_to_keep]
        self.score_column = score_column
        self.label_column = label_column
        self.protected_classes = protected_classes
        self.positive_class_label = positive_class_label
        self.decimals = decimals
        self.date_column = date_column

        # TODO:
        # - Include special values for functions like "thresholds"

    def __str__(self):
        return self.__class__.__name__

    def compute_group_metrics(
        self,
        pre_defined_test: str = "aequitas_group",
        result_wrapper_key: str = "bias",
        include_over_time: bool = True
    ):
        """
        Computes group metrics on a scored & labeled dataset given a protected class.

        Args:
            pre_defined_metrics (str): 'aequitas_group'.

            result_wrapper_key (str): Provides a key to wrap the result dict (flattened results are outside this entry).

            include_over_time (bool): Includes a new section with metrics calculation over a prediction date column.

        Returns:
            Group metrics as computed by some metrics function.
        """

        check_pre_defined_metric(
            pre_defined_test=pre_defined_test,
            metrics_function="compute_group_metrics",
        )

        if pre_defined_test == "aequitas_group":
            result = {result_wrapper_key: []}
            for p_class in self.protected_classes:
                values_over_time = {}
                values = aequitas_group(
                    dataframe=self.dataframe,
                    score_column=self.score_column,
                    label_column=self.label_column,
                    protected_class=p_class,
                    positive_class_label=self.positive_class_label,
                    decimals=self.decimals,
                ).to_dict(orient="records")

                # Change numpy.nan to None if present
                fixed_values = fix_numpy_nans_in_dict_array(dict_array=values)

                group_results = {
                    "test_name": "Aequitas Group",
                    "test_category": "bias",
                    "test_type": "group",
                    "protected_class": p_class["protected_class"],
                    "test_id": "bias_group_{}".format(p_class["protected_class"]),
                    "reference_group": None,
                    "values": fixed_values,
                }
                result[result_wrapper_key].append(group_results)

                if include_over_time and self.date_column is not None:
                    p_class_copy = deepcopy(p_class)
                    p_class_copy["reference_group"] = None
                    values_over_time = self.bias_over_time(aequitas_group, result_wrapper_key + "_group", group_results["test_name"], p_class_copy)

                result.update(values_over_time)

        else:  # place holder for future group methods
            return None

        return result

    def compute_bias_metrics(
        self,
        pre_defined_test: str = "aequitas_bias",
        thresholds: Optional[dict] = None,
        result_wrapper_key: str = "bias",
        flatten: bool = True,
        include_min_max_features: bool = True,
        include_over_time: bool = True
    ):
        """
        Compute bias metrics on a scored & labeled dataset given a protected class and
        a reference group.

        Args:
            pre_defined_metrics (str): 'aequitas_bias'.

            thresholds (dict): a dictionary of threshold values to return with output;
                used later for visualizations. For example, {"min": 0.8, "max": 1.25}.

            result_wrapper_key (str): Provides a key to wrap the result dict (flattened results are outside this entry).

            flatten (bool): Surfaces flat metrics to top level of return dictionary.

            include_min_max_features (bool): Provides the min and max values of the flattened results,
                and their corresponding feature names.
            
            include_over_time (bool): Includes a new section with metrics calculation over a prediction date column.

        Returns:
            Bias metrics as computed by some metrics function.
        """

        check_pre_defined_metric(
            pre_defined_test=pre_defined_test,
            metrics_function="compute_bias_metrics",
        )

        if pre_defined_test == "aequitas_bias":
            result = {result_wrapper_key: []}
            protected_classes = []
            for i, p_class in enumerate(self.protected_classes):
                feature = p_class["protected_class"]
                # If reference_groups not provided, set it as first occurrence in df.protected_class
                r_group = p_class.get("reference_group", None)
                if r_group is None or r_group not in self.dataframe[feature].values:
                    r_groups = self.dataframe[feature].unique()
                    for group in r_groups:
                        # handle unique case of reference group existing in protected classes but not p_class
                        current_group = self.protected_classes[i].get("reference_group", None)
                        if not current_group or current_group not in r_groups:
                            protected_classes.append(
                                {
                                    "protected_class": feature,
                                    "reference_group": group,
                                    "numerical_cutoffs": self.protected_classes[i].get("numerical_cutoffs", None)
                                }
                            )
                        else:
                            protected_classes.append(self.protected_classes[i])
                    logger.warning(
                        "reference_group value %s was not provided or not found. "
                        "Setting to every unique value in data: %s",
                        r_group,
                        feature,
                    )
                else:
                    protected_classes.append(self.protected_classes[i])

            for i, p_class in enumerate(protected_classes):
                values_over_time = {}
                values = aequitas_bias(
                    dataframe=self.dataframe,
                    score_column=self.score_column,
                    label_column=self.label_column,
                    protected_class=p_class,
                    positive_class_label=self.positive_class_label,
                    decimals=self.decimals,
                ).to_dict(orient="records")

                # Change numpy.nan to None if present
                fixed_values = fix_numpy_nans_in_dict_array(dict_array=values)

                bias_results = {
                    "test_name": "Aequitas Bias",
                    "test_category": "bias",
                    "test_type": "bias",
                    "protected_class": p_class["protected_class"],
                    "test_id": "bias_bias_{}".format(p_class["protected_class"]),
                    "reference_group": p_class["reference_group"],
                    "thresholds": thresholds,
                    "values": fixed_values,
                }

                result[result_wrapper_key].append(bias_results)

                if include_over_time and self.date_column is not None:
                    values_over_time = self.bias_over_time(aequitas_group, result_wrapper_key + "_disparity", bias_results["test_name"] + f" - Ref Group {p_class['reference_group']}", p_class)

                result.update(values_over_time)

                if flatten:
                    # Surface top-level metrics
                    for group_dict in bias_results["values"]:
                        result.update(
                            {
                                str(
                                    "ref_"
                                    + p_class["reference_group"]
                                    + "_"
                                    + p_class["protected_class"]
                                    + "_"
                                    + group_dict["attribute_value"]
                                    + "_statistical_parity"
                                ): group_dict["ppr_disparity"],
                                str(
                                    "ref_"
                                    + p_class["reference_group"]
                                    + "_"
                                    + p_class["protected_class"]
                                    + "_"
                                    + group_dict["attribute_value"]
                                    + "_impact_parity"
                                ): group_dict["pprev_disparity"],
                            }
                        )

            # include min and max features for ppr_disparity
            if include_min_max_features:
                first = True
                for k, v in result.items():
                    if "_statistical_parity" not in k:
                        continue
                    if first:
                        max_feature = min_feature = k
                        max_value = min_value = v
                        first = False
                    if v and v > max_value:
                        max_feature = k
                        max_value = v
                    if v and v < min_value:
                        min_feature = k
                        min_value = v
                result["Bias_maxPPRDisparityValue"] = max_value
                result["Bias_maxPPRDisparityValueFeature"] = max_feature
                result["Bias_minPPRDisparityValue"] = min_value
                result["Bias_minPPRDisparityValueFeature"] = min_feature

        else:  # place holder for future bias/disparity methods
            return None

        return result

    def bias_over_time(self, evaluation_function, result_wrapper_key, test_name, p_class) -> dict:
        """
        Computes the bias metrics as given by the evaluation_function, attempting to split the data by date
        :param evaluation_function: The function to run per data split (by date)
        :param result_wrapper_key: The key used for the wrapping the whole test results. This will create a similar key
        :param test_name: The name of the specific test actually run.
        :param p_class: The protected class dictionaries.
        :return: A dictionary with a graph structure over time.
        """
        if self.date_column is not None and self.date_column in self.dataframe:
            df = self.date_index_dataframe(self.dataframe)

            unique_dates = df[self.date_column].unique()

            data = {}
            for date in unique_dates:
                data_of_the_day_sample = df.loc[[date]]
                try:
                    dated_values = evaluation_function(
                        dataframe=data_of_the_day_sample,
                        score_column=self.score_column,
                        label_column=self.label_column,
                        protected_class=p_class,
                        positive_class_label=self.positive_class_label,
                        decimals=self.decimals
                    ).to_dict(orient="records")
                except Exception as err:
                    if isinstance(data_of_the_day_sample, pandas.Series) or data_of_the_day_sample.shape[0] == 1:
                        logger.info(f"Only one record found for day={date}. Metrics cannot be calculated on only one data point. Skipping bias calculation for day={date}.")
                    else:
                        logger.error(str(err) + f". Skipping metrics calculation for day={date}")
                    dated_values = None

                str_date = str(date)
                if dated_values:
                    fixed_values = fix_numpy_nans_in_dict_array(dict_array=dated_values)
                    for attribute in fixed_values:
                        att_name = attribute.pop("attribute_name")
                        att_value = attribute.pop("attribute_value")
                        for metric, value in attribute.items():
                            new_metric = f"{att_name}_{att_value}_{metric}"
                            if new_metric not in data:
                                data[new_metric] = []
                            data[new_metric].append([str_date, value])

            over_time_key = (result_wrapper_key + f"_{att_name}" + "_over_time" +
            (f"_ref_{p_class['reference_group']}" if "reference_group" in p_class and p_class['reference_group'] else ""))
            return {over_time_key: {
                "title": "Bias Over Time" + (" - " + test_name if test_name else ""),
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
        :return: The indexed and sorted dataframe.
        """
        df = df.set_index(check_date_column(df, self.date_column).dt.date)
        df[self.date_column] = check_date_column(df, self.date_column).dt.date
        df = df.sort_index()
        return df

    @staticmethod
    def __make_cutoff_dictionary(numerical_cutoffs: List[float]) -> dict:
        """
        A function to make cutoff buckets for numerical protected class columns.
        Will assume lower bound is -inf and upper bound is +inf.

        Args:
            numerical_cutoffs (List[float]): List of float cutoffs for numerical columns to be bucketed into.

        Returns:
            dict: Dictionary of cutoff integers as keys, buckets as values that are sorted in set notation.
        """

        # Create copy of cutoffs array
        numerical_cutoffs_copy = numerical_cutoffs.copy()

        # Case of singular value in cutoffs list
        if len(numerical_cutoffs) == 1:
            buckets = {}
            buckets[float("-inf")] = f"(-inf, {numerical_cutoffs[0]})"
            buckets[numerical_cutoffs[0]] = f"[{numerical_cutoffs[0]}, +inf)"
            return buckets

        # Sort cutoffs array
        numerical_cutoffs_copy.sort()

        # Create dictionary of bucket values
        buckets = {}
        for i in range(len(numerical_cutoffs_copy) - 1):
            # Place initial bucket of (-inf, starting_value)
            if len(buckets) == 0:
                buckets[float("-inf")] = f"(-inf, {numerical_cutoffs_copy[i]})"
            # Each bucket contains set notation from [i, i+1) of cutoffs
            buckets[
                numerical_cutoffs_copy[i]
            ] = f"[{numerical_cutoffs_copy[i]}, {numerical_cutoffs_copy[i + 1]})"
        # Adding last bucket of [last_value, +inf)
        buckets[numerical_cutoffs_copy[-1]] = f"[{numerical_cutoffs_copy[-1]}, +inf)"
        return buckets

    @staticmethod
    def __label_cutoffs(column: numpy.array, labels: dict) -> List[str]:
        """
        A function that accepts an array-like structure with numerical values
        to be bucketed and outputs the bucket of that value.

        Args:
            column (numpy.array): Column (not column name) of numerical values to be bucketed.
            labels (dict): Cutoff dictionary created through the __make_cutoff_dictionary function.

        Returns:
            List[str]: List of buckets that corresponds to each value in the input column.
        """
        # Create list to hold final buckets
        x_class_list = []

        # Traverse through column
        for value in column:
            # Set initial x_class value
            x_class = None
            # Iterate through the keys of the cutoff dictionary
            for key in labels.keys():
                if value >= key:
                    # Assign appropriate bucket to x_class
                    x_class = labels[key]
            # Apppend to final list
            x_class_list.append(x_class)
        # Return final list, which should be the same length as the input column
        return x_class_list

    @staticmethod
    def __check_and_drop_nulls(
        dataframe: pandas.DataFrame,
        protected_classes: List[dict],
        score_column: str,
        label_column: str,
    ) -> pandas.DataFrame:
        """
        A function to check for and drop NULLs in protected class, score, and label columns.

        Args:
            dataframe (pandas.DataFrame): Pandas DataFrame of scored and labeled data.

            protected_classes (Optional[List[dict]]): List of dictionaries, each defining a protected class,
            of the form {"protected_class": (str) , "reference_group": (str), "numerical_cutoffs": (List)}.

            score_column (str): Column containing predicted values (as computed by underlying model).

            label_column (str): Column containing actual values (ground truths).

        Returns:
            pandas.DataFrame: Copy of self.dataframe with dropped nulls.

        """

        # Checking for NULLs in protected classes
        for p_class in protected_classes:
            protected_class_null_count = (
                dataframe[p_class["protected_class"]].isna().sum()
            )
            if protected_class_null_count > 0:
                logger.warning(
                    "Encounted %i NULL(s) in the protected class column '%s'. Instances of NULL(s) will be dropped prior to Bias and Group Metrics computation.",
                    protected_class_null_count,
                    p_class["protected_class"],
                )
                dataframe = dataframe.dropna(subset=[p_class["protected_class"]])

        # Checking for NULLs in score column
        score_null_count = dataframe[score_column].isna().sum()
        if score_null_count > 0:
            logger.warning(
                "Encountered %i NULL(s) in the score column '%s.' Instances of NULL(s) will be dropped prior to Bias and Group Metrics computation.",
                score_null_count,
                score_column,
            )
            dataframe = dataframe.dropna(subset=[score_column])

        # Checking for NULLs in label column
        label_null_count = dataframe[label_column].isna().sum()
        if label_null_count > 0:
            logger.warning(
                "Encountered %i NULL(s) in the lable column '%s'. Instances of NULL(s) will be dropped prior to Bias and Group Metrics computation.",
                label_null_count,
                label_column,
            )
            dataframe = dataframe.dropna(subset=[label_column])

        return dataframe


def aequitas_group(
    dataframe: pandas.DataFrame,
    score_column: str,
    label_column: str,
    protected_class: dict,
    positive_class_label: Union[int, bool, str],
    decimals: int,
) -> pandas.DataFrame:
    """
    Computes Group metrics using the Aequitas library.

    Args:
        dataframe (pandas.DataFrame): DataFrame containing scores, labels, and protected class.

        score_column (str): name of column containing model predictions (scores).

        label_column (str): name of column containing ground truths (labels).

        protected_class (dict): Dictionary defining a protected class, of the form
            {"protected_class": (str) , "reference_group": (str), "numerical_cutoffs": (List)}.
            "reference_group" is one of the values of protected_class, used for reference.
            "numerical_cutoffs" is a list of float cutoffs for numerical columns to be bucketed into.

        positive_class_label ([int,bool,str]): Binary classification value used to indicate the positive class.

        decimals (int): Number of decimals to round metrics to.

    Returns:
        DataFrame of absolute Group metrics, indexed by the different groups of the
        protected class, e.g., 'male' and 'female'.
    """

    # To measure Bias towards protected_class, filter DataFrame
    # to score, label (ground truth), and protected class
    data_scored = dataframe[
        [score_column, label_column, protected_class["protected_class"]]
    ]

    class_labels = list(data_scored[score_column].unique())
    assert (
        len(class_labels) == 2
    ), "Expected no more than 2 class labels for a binary classification model."
    negative_class_label = (
        class_labels[0] if class_labels[0] != positive_class_label else class_labels[1]
    )

    label_map = {positive_class_label: 1, negative_class_label: 0}

    # Aequitas expects ground truth under 'label_value', and prediction under 'score'
    data_scored = data_scored.rename(
        columns={label_column: "label_value", score_column: "score"}
    )

    data_scored[["label_value", "score"]] = data_scored[
        ["label_value", "score"]
    ].replace(label_map)

    # Process DataFrame
    data_scored_processed, _ = preprocess_input_df(data_scored)

    # Group Metrics
    group = Group()
    xtab, _ = group.get_crosstabs(data_scored_processed)

    # Absolute metrics, such as 'tpr', 'tnr','precision', etc.
    absolute_metrics = group.list_absolute_metrics(xtab)

    # DataFrame of calculated absolute metrics for each sample population group
    absolute_metrics_df = xtab[
        ["attribute_name", "attribute_value"] + absolute_metrics
    ].round(decimals)

    return absolute_metrics_df


def aequitas_bias(
    dataframe: pandas.DataFrame,
    score_column: str,
    label_column: str,
    protected_class: dict,
    positive_class_label: Union[int, bool, str],
    decimals: int,
) -> pandas.DataFrame:
    """
    Computes Bias metrics using the Aequitas library.

    Args:
        dataframe (pandas.DataFrame): DataFrame containing scores, labels, and protected class.

        score_column (str): name of column containing model predictions (scores).

        label_column (str): name of column containing ground truths (labels).

        protected_class (dict): Dictionary defining a protected class, of the form
            {"protected_class": (str) , "reference_group": (str), "numerical_cutoffs": (List)}.
            "reference_group" is one of the values of protected_class, used for reference.
            "numerical_cutoffs" is a list of float cutoffs for numerical columns to be bucketed into.

        positive_class_label ([int,bool,str]): Binary classification value used to indicate the positive class.

        decimals (int): Number of decimals to round metrics to.

    Returns:
        DataFrame of Bias metrics, indexed by the different groups of the protected class,
        e.g., 'male' and 'female'.
    """

    # To measure Bias towards protected_class, filter DataFrame
    # to score, label (ground truth), and protected class
    data_scored = dataframe[
        [score_column, label_column, protected_class["protected_class"]]
    ]

    class_labels = list(data_scored[score_column].unique())
    assert (
        len(class_labels) == 2
    ), "Expected no more than 2 class labels for a binary classification model."
    negative_class_label = (
        class_labels[0] if class_labels[0] != positive_class_label else class_labels[1]
    )

    label_map = {positive_class_label: 1, negative_class_label: 0}

    # Aequitas expects ground truth under 'label_value', and prediction under 'score'
    data_scored = data_scored.rename(
        columns={label_column: "label_value", score_column: "score"}
    )

    data_scored[["label_value", "score"]] = data_scored[
        ["label_value", "score"]
    ].replace(label_map)

    # Process DataFrame
    data_scored_processed, _ = preprocess_input_df(data_scored)

    # Bias Metrics
    bias = Bias()
    group = Group()
    xtab, _ = group.get_crosstabs(data_scored_processed)

    # Disparities calculated in relation <protected_class> for class groups
    bias_df = bias.get_disparity_predefined_groups(
        xtab,
        original_df=data_scored_processed,
        ref_groups_dict={
            protected_class["protected_class"]: protected_class["reference_group"]
        },
    )

    # Disparity metrics added to bias DataFrame
    calculated_disparities = bias.list_disparities(bias_df)

    disparity_metrics_df = bias_df[
        ["attribute_name", "attribute_value"] + calculated_disparities
    ].round(decimals)

    return disparity_metrics_df


if __name__ == "__main__":
    print(doctest.testmod())
    print()
