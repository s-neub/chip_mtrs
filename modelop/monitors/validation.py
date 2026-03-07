"""[summary]

Returns:
    [type]: [description]
"""
import random
import string
from typing import List, Optional

import numpy
import pandas
from statsmodels.stats.weightstats import DescrStatsW as dsw

from modelop.monitors.assertions import (
    check_baseline_and_sample,
    check_columns_in_dataframe,
    check_input_types,
)


class ValidationMonitor:
    """[summary]"""

    def __init__(
        self,
        df_baseline: pandas.DataFrame,
        df_sample: pandas.DataFrame,
        score_column: str,
        label_column: str,
        weight_column: Optional[str] = None,
    ) -> None:

        check_baseline_and_sample(df_baseline=df_baseline, df_sample=df_sample)

        check_input_types(
            inputs=[
                {"label_column": label_column},
                {"score_column": score_column},
            ],
            types=(str),
        )

        assert isinstance(
            weight_column, (str, type(None))
        ), "weight_column should be of type (str) or None."

        check_columns_in_dataframe(
            dataframe=df_baseline, columns=[score_column, label_column]
        )

        if isinstance(weight_column, str):
            assert (
                weight_column in df_baseline.columns
            ), "weight_column does not exist in df_baseline."

        self.score_column = score_column
        self.label_column = label_column
        self.weight_column = weight_column

        train_calc_df = df_baseline.loc[:, [score_column, label_column]]
        eval_calc_df = df_sample.loc[:, [score_column, label_column]]

        chars = string.ascii_uppercase
        weight_var_name = "".join(
            random.choice(chars) for _ in range(6)
        )  # Create a weight variable name that won't clash with any feature or score name
        weight_score_name = "".join(
            random.choice(chars) for _ in range(6)
        )  # Create a weighted score name that won't clash with any feature or score name
        weight_target_name = "".join(
            random.choice(chars) for _ in range(6)
        )  # Create a weighted target name that won't clash with any feature or score name
        weight_non_target_name = "".join(
            random.choice(chars) for _ in range(6)
        )  # Create a weighted non target name that won't clash with any feature or score name

        # Create data frames of train and eval with just the required variables,
        # namely score/feature and weight
        if weight_column is not None:
            train_calc_df.loc[:, weight_var_name] = df_baseline.loc[:, weight_column]
            eval_calc_df.loc[:, weight_var_name] = df_sample.loc[:, weight_column]
        else:
            train_calc_df.loc[:, weight_var_name] = 1.0
            eval_calc_df.loc[:, weight_var_name] = 1.0

        for dataframe in [train_calc_df, eval_calc_df]:

            dataframe.loc[:, weight_score_name] = (
                dataframe.loc[:, weight_var_name] * dataframe.loc[:, score_column]
            )
            dataframe.loc[:, weight_target_name] = (
                dataframe.loc[:, weight_var_name] * dataframe.loc[:, label_column]
            )
            dataframe.loc[:, weight_non_target_name] = (
                dataframe.loc[:, weight_var_name] - dataframe.loc[:, weight_target_name]
            )

        self.train_calc_df = train_calc_df
        self.eval_calc_df = eval_calc_df

        self.weight_var_name = weight_var_name
        self.weight_score_name = weight_score_name
        self.weight_target_name = weight_target_name
        self.weight_non_target_name = weight_non_target_name

    def __str__(self):
        return self.__class__.__name__

    def score_validation_reports(
        self,
        n_lorenz_groups: Optional[int] = 10,
        score_sort_descending: Optional[bool] = True,
        lorenz_group_cuts: Optional[List] = None,
        score_min: Optional[float] = 0.0,
        score_max: Optional[float] = 1.0,
    ) -> tuple:
        """[summary]

        Args:
            n_lorenz_groups (Optional[int], optional): [description]. Defaults to 10.
            score_sort_descending (Optional[bool], optional): [description]. Defaults to True.
            lorenz_group_cuts (Optional[List], optional): [description]. Defaults to [].
            score_min (Optional[float], optional): [description]. Defaults to 0.0.
            score_max (Optional[float], optional): [description]. Defaults to 1.0.

        Returns:
            tuple: [description]
        """
        # Set up empty data frames for lorenz chart
        lorenz_table = pandas.DataFrame(
            {
                "score_range": [],
                "train_count": [],
                "eval_count": [],
                "train_percent": [],
                "eval_percent": [],
                "train_mean_score": [],
                "train_target_rate": [],
                "train_cumulative_target": [],
                "train_cumulative_non_target": [],
                "train_ks_calc": [],
                "train_gini_calc": [],
                "eval_mean_score": [],
                "eval_target_rate": [],
                "eval_cumulative_target": [],
                "eval_cumulative_non_target": [],
                "eval_ks_calc": [],
                "eval_gini_calc": [],
                "train_exp_target": [],
                "train_exp_low_target": [],
                "train_exp_high_target": [],
                "train_actl_target": [],
                "eval_exp_target": [],
                "eval_exp_low_target": [],
                "eval_exp_high_target": [],
                "eval_actl_target": [],
            }
        )

        # Get total number of records, target and non-target (all weighted)
        # for calculating percentages
        total_train_cases = sum(self.train_calc_df.loc[:, self.weight_var_name])
        train_target_cases = sum(self.train_calc_df.loc[:, self.weight_target_name])
        train_non_target_cases = sum(
            self.train_calc_df.loc[:, self.weight_non_target_name]
        )

        total_eval_cases = sum(self.eval_calc_df.loc[:, self.weight_var_name])
        eval_target_cases = sum(self.eval_calc_df.loc[:, self.weight_target_name])
        eval_non_target_cases = sum(
            self.eval_calc_df.loc[:, self.weight_non_target_name]
        )

        # Run the aggregations and calculations for lorenz chart,
        # ks and gini AND Binomial Test expected vs Actual
        if lorenz_group_cuts is None:
            score_cuts = list(
                dsw(
                    self.train_calc_df.loc[:, self.score_column],
                    self.train_calc_df.loc[:, self.weight_var_name],
                ).quantile(probs=numpy.linspace(0, 1, n_lorenz_groups + 1))
            )
            score_cuts[0] = score_min
            score_cuts[len(score_cuts) - 1] = score_max
        else:
            score_cuts = lorenz_group_cuts
            score_cuts = [score_min] + score_cuts + [score_max]

        self.train_calc_df.loc[:, "score_Range"] = pandas.cut(
            self.train_calc_df.loc[:, self.score_column],
            bins=score_cuts,
            duplicates="drop",
            include_lowest=True,
        )
        lorenz_tbl = self.train_calc_df.groupby(["score_Range"]).sum()

        lorenz_tbl.loc[:, "score_range"] = lorenz_tbl.index

        if score_sort_descending:
            j = lorenz_tbl.shape[0] - 1
        else:
            j = 0

        for i in range(lorenz_tbl.shape[0]):
            lorenz_table.loc[i, "score_range"] = lorenz_tbl.loc[j, "score_range"]
            lorenz_table.loc[i, "train_count"] = lorenz_tbl.loc[j, self.weight_var_name]
            lorenz_table.loc[i, "train_percent"] = (
                lorenz_tbl.loc[j, self.weight_var_name] / total_train_cases
            )
            lorenz_table.loc[i, "train_mean_score"] = (
                lorenz_tbl.loc[j, self.weight_score_name]
                / lorenz_table.loc[i, "train_count"]
            )
            lorenz_table.loc[i, "train_target_rate"] = (
                lorenz_tbl.loc[j, self.weight_target_name]
                / lorenz_table.loc[i, "train_count"]
            )
            if i == 0:
                lorenz_table.loc[i, "train_cumulative_target"] = (
                    lorenz_tbl.loc[j, self.weight_target_name] / train_target_cases
                )
                lorenz_table.loc[i, "train_cumulative_non_target"] = (
                    lorenz_tbl.loc[j, self.weight_non_target_name]
                    / train_non_target_cases
                )
                lorenz_table.loc[i, "train_gini_calc"] = (
                    0.5
                    * lorenz_table.loc[i, "train_cumulative_target"]
                    * lorenz_table.loc[i, "train_cumulative_non_target"]
                )
            else:
                lorenz_table.loc[i, "train_cumulative_target"] = (
                    lorenz_table.loc[i - 1, "train_cumulative_target"]
                    + lorenz_tbl.loc[j, self.weight_target_name] / train_target_cases
                )
                lorenz_table.loc[i, "train_cumulative_non_target"] = (
                    lorenz_table.loc[i - 1, "train_cumulative_non_target"]
                    + lorenz_tbl.loc[j, self.weight_non_target_name]
                    / train_non_target_cases
                )
                lorenz_table.loc[i, "train_gini_calc"] = (
                    0.5
                    * (
                        lorenz_table.loc[i, "train_cumulative_target"]
                        + lorenz_table.loc[i - 1, "train_cumulative_target"]
                    )
                    * (
                        lorenz_table.loc[i, "train_cumulative_non_target"]
                        - lorenz_table.loc[i - 1, "train_cumulative_non_target"]
                    )
                )
            lorenz_table.loc[i, "train_ks_calc"] = numpy.abs(
                lorenz_table.loc[i, "train_cumulative_target"]
                - lorenz_table.loc[i, "train_cumulative_non_target"]
            )

            lorenz_table.loc[i, "train_exp_target"] = (
                lorenz_table.loc[i, "train_count"]
                * lorenz_table.loc[i, "train_mean_score"]
            )
            lorenz_table.loc[i, "train_exp_low_target"] = lorenz_table.loc[
                i, "train_exp_target"
            ] - 2 * numpy.sqrt(
                lorenz_table.loc[i, "train_count"]
                * lorenz_table.loc[i, "train_mean_score"]
                * (1.0 - lorenz_table.loc[i, "train_mean_score"])
            )
            lorenz_table.loc[i, "train_exp_high_target"] = lorenz_table.loc[
                i, "train_exp_target"
            ] + 2 * numpy.sqrt(
                lorenz_table.loc[i, "train_count"]
                * lorenz_table.loc[i, "train_mean_score"]
                * (1.0 - lorenz_table.loc[i, "train_mean_score"])
            )
            lorenz_table.loc[i, "train_actl_target"] = lorenz_tbl.loc[
                j, self.weight_target_name
            ]
            j = j - 1

        self.eval_calc_df.loc[:, "score_Range"] = pandas.cut(
            self.eval_calc_df.loc[:, self.score_column],
            bins=score_cuts,
            duplicates="drop",
            include_lowest=True,
        )
        lorenz_tbl = self.eval_calc_df.groupby(["score_Range"]).sum()

        lorenz_tbl.loc[:, "score_range"] = lorenz_tbl.index

        if score_sort_descending:
            j = lorenz_tbl.shape[0] - 1
        else:
            j = 0

        for i in range(lorenz_tbl.shape[0]):
            lorenz_table.loc[i, "score_range"] = lorenz_tbl.loc[j, "score_range"]
            lorenz_table.loc[i, "eval_count"] = lorenz_tbl.loc[j, self.weight_var_name]
            lorenz_table.loc[i, "eval_percent"] = (
                lorenz_tbl.loc[j, self.weight_var_name] / total_eval_cases
            )
            lorenz_table.loc[i, "eval_mean_score"] = (
                lorenz_tbl.loc[j, self.weight_score_name]
                / lorenz_table.loc[i, "eval_count"]
            )
            lorenz_table.loc[i, "eval_target_rate"] = (
                lorenz_tbl.loc[j, self.weight_target_name]
                / lorenz_table.loc[i, "eval_count"]
            )
            if i == 0:
                lorenz_table.loc[i, "eval_cumulative_target"] = (
                    lorenz_tbl.loc[j, self.weight_target_name] / eval_target_cases
                )
                lorenz_table.loc[i, "eval_cumulative_non_target"] = (
                    lorenz_tbl.loc[j, self.weight_non_target_name]
                    / eval_non_target_cases
                )
                lorenz_table.loc[i, "eval_gini_calc"] = (
                    0.5
                    * lorenz_table.loc[i, "eval_cumulative_target"]
                    * lorenz_table.loc[i, "eval_cumulative_non_target"]
                )
            else:
                lorenz_table.loc[i, "eval_cumulative_target"] = (
                    lorenz_table.loc[i - 1, "eval_cumulative_target"]
                    + lorenz_tbl.loc[j, self.weight_target_name] / eval_target_cases
                )
                lorenz_table.loc[i, "eval_cumulative_non_target"] = (
                    lorenz_table.loc[i - 1, "eval_cumulative_non_target"]
                    + lorenz_tbl.loc[j, self.weight_non_target_name]
                    / eval_non_target_cases
                )
                lorenz_table.loc[i, "eval_gini_calc"] = (
                    0.5
                    * (
                        lorenz_table.loc[i, "eval_cumulative_target"]
                        + lorenz_table.loc[i - 1, "eval_cumulative_target"]
                    )
                    * (
                        lorenz_table.loc[i, "eval_cumulative_non_target"]
                        - lorenz_table.loc[i - 1, "eval_cumulative_non_target"]
                    )
                )
            lorenz_table.loc[i, "eval_ks_calc"] = numpy.abs(
                lorenz_table.loc[i, "eval_cumulative_target"]
                - lorenz_table.loc[i, "eval_cumulative_non_target"]
            )

            lorenz_table.loc[i, "eval_exp_target"] = (
                lorenz_table.loc[i, "eval_count"]
                * lorenz_table.loc[i, "eval_mean_score"]
            )
            lorenz_table.loc[i, "eval_exp_low_target"] = lorenz_table.loc[
                i, "eval_exp_target"
            ] - 2 * numpy.sqrt(
                lorenz_table.loc[i, "eval_count"]
                * lorenz_table.loc[i, "eval_mean_score"]
                * (1.0 - lorenz_table.loc[i, "eval_mean_score"])
            )
            lorenz_table.loc[i, "eval_exp_high_target"] = lorenz_table.loc[
                i, "eval_exp_target"
            ] + 2 * numpy.sqrt(
                lorenz_table.loc[i, "eval_count"]
                * lorenz_table.loc[i, "eval_mean_score"]
                * (1.0 - lorenz_table.loc[i, "eval_mean_score"])
            )
            lorenz_table.loc[i, "eval_actl_target"] = lorenz_tbl.loc[
                j, self.weight_target_name
            ]
            j = j - 1

        lorenz_dict = {
            "train_ks": max(lorenz_table.loc[:, "train_ks_calc"]),
            "train_gini": 2.0 * sum(lorenz_table.loc[:, "train_gini_calc"]) - 1.0,
            "eval_ks": max(lorenz_table.loc[:, "eval_ks_calc"]),
            "eval_gini": 2.0 * sum(lorenz_table.loc[:, "eval_gini_calc"]) - 1.0,
            "lorenz_table": lorenz_table,
        }

        # Generating dataframe for score pointwise ks plot, score concentration and selection curve
        train_df_pointwise = (
            self.train_calc_df.groupby([self.score_column])
            .sum()
            .loc[
                :,
                [
                    self.weight_var_name,
                    self.weight_target_name,
                    self.weight_non_target_name,
                ],
            ]
        )
        train_df_pointwise.rename(
            columns={
                self.weight_var_name: "train_num_cases",
                self.weight_target_name: "train_num_target",
                self.weight_non_target_name: "train_num_non_target",
            },
            inplace=True,
        )
        eval_df_pointwise = (
            self.eval_calc_df.groupby([self.score_column])
            .sum()
            .loc[
                :,
                [
                    self.weight_var_name,
                    self.weight_target_name,
                    self.weight_non_target_name,
                ],
            ]
        )
        eval_df_pointwise.rename(
            columns={
                self.weight_var_name: "eval_num_cases",
                self.weight_target_name: "eval_num_target",
                self.weight_non_target_name: "eval_num_non_target",
            },
            inplace=True,
        )

        pointwise_df = pandas.merge(
            train_df_pointwise, eval_df_pointwise, how="outer", on=[self.score_column]
        )

        for source in ["train", "eval"]:
            pointwise_df.loc[
                pandas.isna(pointwise_df[source + "_num_cases"]),
                (source + "_num_cases"),
            ] = 0
            pointwise_df.loc[
                pandas.isna(pointwise_df[source + "_num_target"]),
                (source + "_num_target"),
            ] = 0
            pointwise_df.loc[
                pandas.isna(pointwise_df[source + "_num_non_target"]),
                (source + "_num_non_target"),
            ] = 0

        if score_sort_descending:
            pointwise_df.sort_values(
                by=[self.score_column], ascending=False, inplace=True
            )

        total_cases = total_train_cases
        target_cases = train_target_cases
        non_target_cases = train_non_target_cases
        for source in ["train", "eval"]:

            pointwise_df.loc[:, source + "_perc_cases"] = (
                pointwise_df.loc[:, source + "_num_cases"] / total_cases
            )

            pointwise_df.loc[:, source + "_selection_rate"] = (
                1.0 - pointwise_df.loc[:, source + "_perc_cases"].cumsum()
            )

            pointwise_df.loc[:, source + "_target_rate"] = (
                target_cases - pointwise_df.loc[:, source + "_num_target"].cumsum()
            ) / (total_cases - pointwise_df.loc[:, source + "_num_cases"].cumsum())

            pointwise_df.loc[:, source + "_cumperc_target"] = (
                pointwise_df.loc[:, source + "_num_target"].cumsum() / target_cases
            )
            pointwise_df.loc[:, source + "_cumperc_non_target"] = (
                pointwise_df.loc[:, source + "_num_non_target"].cumsum()
                / non_target_cases
            )
            total_cases = total_eval_cases
            target_cases = eval_target_cases
            non_target_cases = eval_non_target_cases

        (pointwise_ks_train, pointwise_ks_eval) = [
            max(
                numpy.abs(
                    pointwise_df.loc[:, source + "_cumperc_target"]
                    - pointwise_df.loc[:, source + "_cumperc_non_target"]
                )
            )
            for source in ["train", "eval"]
        ]
        (pointwise_gini_train, pointwise_gini_eval) = [
            (
                2.0
                * sum(
                    0.5
                    * (
                        pointwise_df.loc[:, source + "_cumperc_target"]
                        + (
                            [0]
                            + list(
                                pointwise_df.iloc[
                                    0 : pointwise_df.shape[0] - 1,
                                    pointwise_df.columns.get_loc(
                                        source + "_cumperc_target"
                                    ),
                                ]
                            )
                        )
                    )
                    * (
                        pointwise_df.loc[:, source + "_cumperc_non_target"]
                        - (
                            [0]
                            + list(
                                pointwise_df.iloc[
                                    0 : pointwise_df.shape[0] - 1,
                                    pointwise_df.columns.get_loc(
                                        source + "_cumperc_non_target"
                                    ),
                                ]
                            )
                        )
                    )
                )
                - 1.0
            )
            for source in ["train", "eval"]
        ]

        score_level_dict = {
            "train_scoreLevel_ks": pointwise_ks_train,
            "train_scoreLevel_gini": pointwise_gini_train,
            "eval_scoreLevel_ks": pointwise_ks_eval,
            "eval_scoreLevel_gini": pointwise_gini_eval,
            "dataframe_score_validation_plots": pointwise_df,
        }

        return (lorenz_dict, score_level_dict)
