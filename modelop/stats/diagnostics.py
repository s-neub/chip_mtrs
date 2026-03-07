"""This module provides several tests for regression models.

See `BiasMonitor` for usage examples.

.. toctree::

.. autosummary::
   :toctree: _autosummary
"""
import doctest
import logging
from typing import List, Optional
import numpy.ma as ma

# Third party packages
import numpy
import pandas
import statsmodels
from scipy.stats import cramervonmises
try:
    from scipy.stats.stats import PearsonRConstantInputWarning as ConstantInputWarning
except ImportError:
    from scipy.stats import ConstantInputWarning
from scipy.stats.mstats import pearsonr
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_breuschpagan,
    kstest_normal,
    linear_lm,
    normal_ad,
)
from statsmodels.stats.stattools import durbin_watson

import modelop.schema.infer as infer
from modelop.monitors.assertions import check_columns_in_dataframe, check_date_column, check_input_types
from modelop.utils import (
    fix_numpy_nans_in_dict,
    fix_numpy_nans_in_dict_array,
    get_min_max_values_keys_from_dict,
)

pandas.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)

# @Global


class HomoscedasticityMetrics:
    """
    Class to compute Homoscedasticity Metrics.

    Args:
        dataframe (pandas.DataFrame): Input DataFrame.

        job_json (dict): JSON dictionary with the metadata of the model.

        score_column (str): Model output.

        label_column (str): Ground truth.

        numerical_predictors (Optional[list], optional): Predictor variables. Defaults to None.

        date_column (str): Column containing dates for over time metrics.

        decimals (Optional[int], optional): Roundign parameter. Defaults to 4.

    Examples:
        Load 'boston house prices' dataset and compute homoscedasticity metrics:

        >>> import pandas
        >>> from pprint import pprint
        >>> from sklearn.datasets import load_boston

        >>> # load data
        >>> boston = load_boston()
        >>> X = pandas.DataFrame(boston.data, columns=boston.feature_names)
        >>> X.drop('CHAS', axis=1, inplace=True)
        >>> y = pandas.Series(boston.target, name='MEDV')

        >>> # Run linear regression
        >>> import statsmodels.api as sm
        >>> X_constant = sm.add_constant(X)
        >>> lin_reg = sm.OLS(y,X_constant).fit(random=0)

        >>> # Get predictions
        >>> X_constant["MEDV_pred"] = lin_reg.predict(X_constant).round(1)
        >>> # Add ground truth to DataFrame
        >>> X_constant["MEDV"] = y

        >>> from modelop.stats.diagnostics import HomoscedasticityMetrics
        >>> homoscedasticity_metrics = HomoscedasticityMetrics(
        ...     dataframe=X_constant,
        ...     score_column="MEDV_pred",
        ...     label_column="MEDV",
        ...     numerical_predictors=[
        ...         'const', 'CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
        ...         'TAX', 'PTRATIO', 'B', 'LSTAT'
        ...     ]
        ... )

        >>> pprint(
        ...     homoscedasticity_metrics.breusch_pagan_test(
        ...         flatten=False),
        ...     sort_dicts=False
        ... )
        {'homoscedasticity': [{'test_name': 'Breusch-Pagan',
                               'test_category': 'homoscedasticity',
                               'test_type': 'breusch_pagan',
                               'test_id': 'homoscedasticity_breusch_pagan',
                               'values': {'lm_statistic': 60.0945,
                                          'lm_p_value': 0.0,
                                          'f_statistic': 5.5368,
                                          'f_p_value': 0.0}}]}

        >>> pprint(
        ...     homoscedasticity_metrics.engle_lagrange_multiplier_test(
        ...         flatten=False),
        ...     sort_dicts=False
        ... )
        {'homoscedasticity': [{'test_name': "Engle's Lagrange Multiplier",
                               'test_category': 'homoscedasticity',
                               'test_type': 'engle_lagrange_multiplier',
                               'test_id': 'homoscedasticity_engle_lagrange_multiplier',
                               'values': {'lm_statistic': 174.444, 'lm_p_value': 0.0}}]}

        >>> pprint(
        ...     homoscedasticity_metrics.ljung_box_q_test(
        ...         include_min_max_features=True
        ...     ),
        ...     sort_dicts=False
        ... )
        {'Homoscedasticity_minLjungBoxQPValue': 0.0,
         'Homoscedasticity_minLjungBoxQPValueFeature': 'lag_1',
         'Homoscedasticity_maxLjungBoxQPValue': 0.0,
         'Homoscedasticity_maxLjungBoxQPValueFeature': 'lag_1',
         'homoscedasticity': [{'test_name': 'Ljung-Box Q',
                               'test_category': 'homoscedasticity',
                               'test_type': 'ljung_box_q',
                               'test_id': 'homoscedasticity_ljung_box_q',
                               'values': [{'lag': 1,
                                           'lb_statistic': 119.549,
                                           'lb_p_value': 0.0},
                                          {'lag': 2,
                                           'lb_statistic': 179.9276,
                                           'lb_p_value': 0.0},
                                          {'lag': 3,
                                           'lb_statistic': 220.8348,
                                           'lb_p_value': 0.0},
                                          {'lag': 4,
                                           'lb_statistic': 250.7179,
                                           'lb_p_value': 0.0},
                                          {'lag': 5,
                                           'lb_statistic': 265.4203,
                                           'lb_p_value': 0.0},
                                          {'lag': 6,
                                           'lb_statistic': 267.5502,
                                           'lb_p_value': 0.0},
                                          {'lag': 7,
                                           'lb_statistic': 269.6406,
                                           'lb_p_value': 0.0},
                                          {'lag': 8,
                                           'lb_statistic': 273.8128,
                                           'lb_p_value': 0.0},
                                          {'lag': 9,
                                           'lb_statistic': 275.1769,
                                           'lb_p_value': 0.0},
                                          {'lag': 10,
                                           'lb_statistic': 279.4712,
                                           'lb_p_value': 0.0}]}]}
    """

    def __init__(
        self,
        dataframe: pandas.DataFrame,
        job_json: dict = None,
        score_column: str = None,
        label_column: str = None,
        numerical_predictors: Optional[list] = None,
        date_column: str = None,
        decimals: Optional[int] = 4,
        **kwargs,
    ) -> None:

        assert isinstance(
            dataframe, pandas.DataFrame
        ), "dataframe should be of type (pandas.DataFrame)."

        if job_json is not None:
            logger.info(
                "Parameter 'job_json' is present and will be used to extract "
                "'score_column', 'label_column', and 'numerical_predictors'."
            )
            input_schema_definition = infer.extract_input_schema(job_json)
            monitoring_parameters = infer.set_monitoring_parameters(
                schema_json=input_schema_definition, check_schema=True
            )
            score_column = monitoring_parameters["score_column"]
            label_column = monitoring_parameters["label_column"]
            date_column = monitoring_parameters["date_column"]
            numerical_predictors = monitoring_parameters["numerical_columns"]
        else:
            if (
                score_column is None
                or label_column is None
                or numerical_predictors is None
            ):
                missing_args_error = (
                    "Parameter 'job_json' is not present, "
                    "but one of 'score_column', 'label_column', or 'numerical_predictors' was not provided. "
                    "'score_column', 'label_column', and 'numerical_predictors' input parameters are "
                    "required if 'job_json' is not provided."
                )
                logger.error(missing_args_error)
                raise Exception(missing_args_error)

        check_input_types(
            inputs=[
                {"numerical_predictors": numerical_predictors},
            ],
            types=(list, type(None)),
        )

        check_input_types(
            inputs=[{"score_column": score_column}, {"label_column": label_column}],
            types=(str),
        )

        if date_column is not None:
            check_input_types(inputs=[{"date_column": date_column}], types=(str))

        if date_column is not None:
            check_columns_in_dataframe(
                dataframe=dataframe, columns=[score_column, label_column, date_column]
            )
        else:
            check_columns_in_dataframe(
                dataframe=dataframe, columns=[score_column, label_column]
            )

        if numerical_predictors is not None:
            check_columns_in_dataframe(
                dataframe=dataframe, columns=numerical_predictors
            )

        # Infer numerical predictors if not specified
        else:
            numerical_predictors = [
                col
                for col in dataframe.columns
                if (
                    numpy.isin(
                        dataframe.dtypes[col],
                        ["float32", "float64", "int32", "int64", "uint8"],
                    )
                    and not numpy.isin(col, [label_column, score_column])
                )
            ]
            print("Identified numerical predictor(s): ", numerical_predictors)

        assert (
            len(numerical_predictors) > 0
        ), "dataframe must contain at least 1 numerical_predictor column!"

        # Restricting dataframe to columns of interest
        if date_column is not None:
            dataframe = dataframe[numerical_predictors + [score_column, label_column, date_column]]
        else:
            dataframe = dataframe[numerical_predictors + [score_column, label_column]]

        # Run reordering of columns
        self.dataframe = handle_constant_columns(
            dataframe,
        )
        self.numerical_predictors = list(self.dataframe.columns)
        self.numerical_predictors.remove(label_column)
        self.numerical_predictors.remove(score_column)
        if date_column is not None and date_column in self.numerical_predictors:
            self.numerical_predictors.remove(date_column)

        assert (
            self.dataframe[self.numerical_predictors[0]].nunique() == 1
        ), "First column in dataframe must be constant! See https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html and https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.linear_lm.html"
        self.label_column = label_column
        self.score_column = score_column
        self.date_column = date_column
        self.residuals = dataframe[label_column] - dataframe[score_column]
        self.exog = dataframe[numerical_predictors]
        self.decimals = decimals
        self.result = None

    def __str__(self):
        return self.__class__.__name__

    def breusch_pagan_test(
        self,
        robust: bool = True,
        result_wrapper_key: str = "homoscedasticity",
        flatten: bool = True,
        include_over_time: bool = True
    ) -> dict:

        """
        Breusch-Pagan Lagrange Multiplier test for heteroscedasticity.

        Args:
            robust (bool, optional): Flag indicating whether to use the Koenker version of the
                test (default) which assumes independent and identically distributed error terms,
                or the original Breusch-Pagan version which assumes residuals are normally
                distributed. Defaults to True.

            result_wrapper_key (str): Provides a key to wrap the result dict (flattened results are outside this entry).

            flatten (bool): Surfaces flat metrics to top level of return dictionary.

            include_over_time (bool): Includes a new section with metrics calculation over a prediction date column.

        Returns:
            dict: Test results, including:
                lm_statistic (float): Lagrange multiplier test statistic
                lm_p_value (float): p-value of Lagrange multiplier test
                f_statistic (float): f-statistic of the hypothesis that the error variance does not depend on x
                f_p_value (float): p-value for the f-statistic

        Notes:
            Built on: statsmodels>=0.12.2,<=0.13.2
            https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html
        """
        values_over_time = {}
        test_results = {
            "test_name": "Breusch-Pagan",
            "test_category": "homoscedasticity",
            "test_type": "breusch_pagan",
            "test_id": "homoscedasticity_breusch_pagan",
            "values": self.__calculate_breusch_pagan(robust),
        }

        if include_over_time and self.date_column is not None:
            values_over_time = diagnostics_over_time(
                self,
                self.__calculate_breusch_pagan,
                result_wrapper_key=result_wrapper_key,
                test_name="Breusch-Pagan",
                test_title="Homoscedasticity",
                custom_args={
                    "robust": robust
                }
            )
        self.result = values_over_time if values_over_time else {}
        if flatten:
            self.result.update({
                # Top-level metrics
                "breusch_pagan_lm_statistic": test_results["values"]["lm_statistic"],
                "breusch_pagan_lm_p_value": test_results["values"]["lm_p_value"],
                "breusch_pagan_f_statistic": test_results["values"]["f_statistic"],
                "breusch_pagan_f_p_value": test_results["values"]["f_p_value"],
                # Vanilla HomoscedasticityMetrics output
                result_wrapper_key: [test_results],
            })
        else:
            self.result.update({result_wrapper_key: [test_results]})
        return self.result
    
    def __calculate_breusch_pagan(self, robust) -> dict:
        """
        Breusch-Pagan Lagrange Multiplier test for heteroscedasticity.
        """
        values = het_breuschpagan(
            # resid : array_like. This should be the residual of a regression.
            resid=self.residuals,
            # exog_het : array_like. Contains variables suspected of being related to heteroscedasticity.
            exog_het=self.exog,
            robust=robust,
        )

        values_dict = {
            "lm_statistic": numpy.round(values[0], self.decimals),
            "lm_p_value": numpy.round(values[1], self.decimals),
            "f_statistic": numpy.round(values[2], self.decimals),
            "f_p_value": numpy.round(values[3], self.decimals),
        }

        # Change numpy.nan to None if present
        return fix_numpy_nans_in_dict(dictionary=values_dict)

    def engle_lagrange_multiplier_test(
        self,
        func=None,
        result_wrapper_key: str = "homoscedasticity",
        flatten: bool = True,
        include_over_time: bool = True
    ) -> dict:
        """
        Lagrange multiplier test for linearity against functional alternative.

        Args:
            func (callable, optional): If func is None, then squares are used. func needs to take
            an array of exog and return an array of transformed variables. Defaults to None.

            result_wrapper_key (str): Provides a key to wrap the result dict (flattened results are outside this entry).

            flatten (bool): Surfaces flat metrics to top level of return dictionary.

            include_over_time (bool): Includes a new section with metrics calculation over a prediction date column.

        Returns:
            dict: Test results, including:
                lm_statistic (float): Lagrange multiplier test statistic
                lm_p_value (float): p-value of Lagrange multiplier test

        Notes:
            Built on: statsmodels>=0.12.2,<=0.13.2
            https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.linear_lm.html
        """
        values_over_time = {}
        test_results = {
            "test_name": "Engle's Lagrange Multiplier",
            "test_category": "homoscedasticity",
            "test_type": "engle_lagrange_multiplier",
            "test_id": "homoscedasticity_engle_lagrange_multiplier",
            "values": self.__calculate_engle_lagrange_multiplier(func),
        }

        if include_over_time and self.date_column is not None:
            values_over_time = diagnostics_over_time(
                self,
                self.__calculate_engle_lagrange_multiplier,
                result_wrapper_key=result_wrapper_key,
                test_name="Engle's Lagrange Multiplier",
                test_title="Homoscedasticity",
                custom_args={
                    "func": func
                }
            )
        self.result = values_over_time if values_over_time else {}
        if flatten:
            self.result.update({
                # Top-level metrics
                "engle_lm_statistic": test_results["values"]["lm_statistic"],
                "engle_lm_p_value": test_results["values"]["lm_p_value"],
                # Vanilla HomoscedasticityMetrics output
                result_wrapper_key: [test_results],
            })
        else:
            self.result.update({result_wrapper_key: [test_results]})
        return self.result
    
    def __calculate_engle_lagrange_multiplier(self, func) -> dict:
        """
        Lagrange multiplier test for linearity against functional alternative.
        """
        values = linear_lm(
            # resid : ndarray. Residuals of a regression
            resid=numpy.array(self.residuals),
            # exog : ndarray. Exogenous variables for which linearity is tested
            exog=numpy.array(self.exog),
            func=func,
        )

        values_dict = {
            "lm_statistic": numpy.round(values[0], self.decimals),
            "lm_p_value": numpy.round(values[1], self.decimals),
            # "f-test": values[2] # Will ignore F-test variant statistic and p-value for now as it returns a string
        }

        # Change numpy.nan to None if present
        return fix_numpy_nans_in_dict(dictionary=values_dict)

    def ljung_box_q_test(
        self,
        lags=None,
        boxpierce=False,
        model_df=0,
        period=None,
        auto_lag=False,
        return_df=True,
        result_wrapper_key="homoscedasticity",
        include_min_max_features=True,
    ) -> dict:
        """
        Ljung-Box test of autocorrelation in residuals.

        Args:
            lags ({int, array_like}, optional): If lags is an integer then this is taken to be the
                largest lag that is included, the test result is reported for all smaller lag
                length. If lags is a list or array, then all lags are included up to the largest
                lag in the list, however only the tests for the lags in the list are reported. If
                lags is None, then the default maxlag is min(10, nobs // 5). The default number of
                lags changes if period is set. Defaults to None.

            boxpierce (bool, optional): If true, then additional to the results of the Ljung-Box
                test also the Box-Pierce test results are returned. Defaults to False.

            model_df (int, optional): Number of degrees of freedom consumed by the model. In an
                ARMA model, this value is usually p+q where p is the AR order and q is the MA
                order. This value is subtracted from the degrees-of-freedom used in the test so
                that the adjusted dof for the statistics are lags - model_df.
                If lags - model_df <= 0, then NaN is returned. Defaults to 0.

            period (int, optional): The period of a Seasonal time series. Used to compute the max
                lag for seasonal data which uses min(2*period, nobs // 5) if set. If None, then the
                default rule is used to set the number of lags. When set, must be >= 2. Defaults to
                None.

            auto_lag (bool, optional): Flag indicating whether to automatically determine the
                optimal lag length based on threshold of maximum correlation value. Defaults to
                False

            result_wrapper_key (str): Provides a key to wrap the result dict (flattened results are outside this entry).

            include_min_max_features (bool): Provides the min and max values of the results,
                and their corresponding lags.

        Returns:
            dict: Test results, including:
                lb_statistic (float): Ljung-Box test statistic
                lb_p_value (float): The p-value based on chi-square distribution. The p-value is
                    computed as 1.0 - chi2.cdf(lbvalue, dof) where dof is lag - model_df.
                    If lag - model_df <= 0, then NaN is returned for the pvalue.

        Notes:
            Built on: statsmodels==0.13.2
            https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html
        """
        fixed_values = self.__calculate_ljung_box_q(
                    lags=lags,
                    boxpierce=boxpierce,
                    model_df=model_df,
                    period=period,
                    auto_lag=auto_lag,
                    return_df=return_df
                )
        test_results = {
            "test_name": "Ljung-Box Q",
            "test_category": "homoscedasticity",
            "test_type": "ljung_box_q",
            "test_id": "homoscedasticity_ljung_box_q",
            "values": fixed_values
        }

        self.result = {}

        if include_min_max_features:

            pval_lag_dict = {i["lag"]: i["lb_p_value"] for i in fixed_values}
            min_max_dict = get_min_max_values_keys_from_dict(values_dict=pval_lag_dict)

            self.result["Homoscedasticity_minLjungBoxQPValue"] = min_max_dict[
                "min_value"
            ]
            self.result["Homoscedasticity_minLjungBoxQPValueFeature"] = "lag_" + str(
                min_max_dict["min_feature"]
            )
            self.result["Homoscedasticity_maxLjungBoxQPValue"] = min_max_dict[
                "max_value"
            ]
            self.result["Homoscedasticity_maxLjungBoxQPValueFeature"] = "lag_" + str(
                min_max_dict["max_feature"]
            )

        # Vanilla HomoscedasticityMetrics output
        self.result[result_wrapper_key] = [test_results]
        return self.result

    def __calculate_ljung_box_q(
        self,
        lags=None,
        boxpierce=False,
        model_df=0,
        period=None,
        auto_lag=False,
        return_df=True
    ) -> dict:
        """
        Ljung-Box test of autocorrelation in residuals.
        """
        assert (
            statsmodels.__version__ == "0.13.2"
        ), "This method requires statsmodels== 0.13.2."

        values = acorr_ljungbox(
            # x: array_like. The data series.
            x=numpy.array(self.residuals),
            lags=lags,
            boxpierce=boxpierce,
            model_df=model_df,
            period=period,
            auto_lag=auto_lag,
            return_df=return_df,
        )

        # Set lag value column
        if isinstance(lags, (int, type(None))):
            values["lag"] = values.index
        elif isinstance(lags, list):
            values["lag"] = lags

        values = values.rename(
            columns={"lb_stat": "lb_statistic", "lb_pvalue": "lb_p_value"}
        )

        values = values[["lag", "lb_statistic", "lb_p_value"]].round(self.decimals)

        # Change numpy.nan to None if present
        fixed_values = fix_numpy_nans_in_dict_array(
            dict_array=values.to_dict(orient="records")
        )
        return fixed_values

class NormalityMetrics:
    """
    Class to compute metrics related to Normality.

    Args:
        dataframe (pandas.DataFrame): Input DataFrame.

        job_json (dict): JSON dictionary with the metadata of the model.

        score_column (str): Model output.

        label_column (str): Ground truth.

        decimals (Optional[int], optional): Rounding parameter. Defaults to 4.

        date_column (str): Column containing dates for over time metrics.

    Examples:
        Load 'boston house prices' dataset and compute normality metrics:

        >>> import pandas
        >>> from pprint import pprint
        >>> from sklearn.datasets import load_boston

        >>> # load data
        >>> boston = load_boston()
        >>> X = pandas.DataFrame(boston.data, columns=boston.feature_names)
        >>> X.drop('CHAS', axis=1, inplace=True)
        >>> y = pandas.Series(boston.target, name='MEDV')

        >>> # Run linear regression
        >>> import statsmodels.api as sm
        >>> X_constant = sm.add_constant(X)
        >>> lin_reg = sm.OLS(y,X_constant).fit(random=0)

        >>> # Get predictions
        >>> X_constant["MEDV_pred"] = lin_reg.predict(X_constant).round(1)
        >>> # Add ground truth to DataFrame
        >>> X_constant["MEDV"] = y

        >>> from modelop.stats.diagnostics import NormalityMetrics
        >>> normality_metrics = NormalityMetrics(
        ...     dataframe=X_constant,
        ...     score_column="MEDV_pred",
        ...     label_column="MEDV",
        ... )

        >>> pprint(
        ...     normality_metrics.kolmogorov_smirnov_test(
        ...         flatten=False
        ...     ),
        ...     sort_dicts=False
        ... )
        {'normality': [{'test_name': 'Kolmogorov-Smirnov',
                        'test_category': 'normality',
                        'test_type': 'kolmogorov_smirnov',
                        'test_id': 'normality_kolmogorov_smirnov',
                        'values': {'ks_statistic': 0.1284, 'ks_p_value': 0.001}}]}

        >>> pprint(
        ...     normality_metrics.anderson_darling_test(
        ...         flatten=True
        ...     ),
        ...     sort_dicts=False
        ... )
        {'ad_statistic': 10.9366,
         'ad_p_value': 0.0,
         'normality': [{'test_name': 'Anderson-Darling',
                        'test_category': 'normality',
                        'test_type': 'anderson_darling',
                        'test_id': 'normality_anderson_darling',
                        'values': {'ad_statistic': 10.9366, 'ad_p_value': 0.0}}]}

        >>> pprint(
        ...     normality_metrics.cramer_von_mises_test(
        ...         flatten=True
        ...     ),
        ...     sort_dicts=False
        ... )
        {'cvm_statistic': 18.4655,
         'cvm_p_value': 0.0,
         'normality': [{'test_name': 'Cramer-von Mises',
                        'test_category': 'normality',
                        'test_type': 'cramer_von_mises',
                        'test_id': 'normality_cramer_von_mises',
                        'values': {'cvm_statistic': 18.4655, 'cvm_p_value': 0.0}}]}
    """

    def __init__(
        self,
        dataframe: pandas.DataFrame,
        job_json: dict = None,
        score_column: str = None,
        label_column: str = None,
        decimals: Optional[int] = 4,
        date_column: str = None,
    ) -> None:

        assert isinstance(
            dataframe, pandas.DataFrame
        ), "dataframe should be of type (pandas.DataFrame)."

        if job_json is not None:
            logger.info(
                "Parameter 'job_json' is present and will be used to extract "
                "'score_column' and 'label_column'."
            )
            input_schema_definition = infer.extract_input_schema(job_json)
            monitoring_parameters = infer.set_monitoring_parameters(
                schema_json=input_schema_definition, check_schema=True
            )
            score_column = monitoring_parameters["score_column"]
            label_column = monitoring_parameters["label_column"]
            date_column = monitoring_parameters["date_column"]
        else:
            if score_column is None or label_column is None:
                missing_args_error = (
                    "Parameter 'job_json' is not present, "
                    "but one of 'score_column' or 'label_column' was not provided. "
                    "'score_column' and 'label_column' input parameters are "
                    "required if 'job_json' is not provided."
                )
                logger.error(missing_args_error)
                raise Exception(missing_args_error)

        check_input_types(
            inputs=[{"score_column": score_column}, {"label_column": label_column}],
            types=(str),
        )

        if date_column is not None:
            check_input_types(inputs=[{"date_column": date_column}], types=(str))

        if date_column is not None:
            check_columns_in_dataframe(
                dataframe=dataframe, columns=[score_column, label_column, date_column]
            )
        else:
            check_columns_in_dataframe(
                dataframe=dataframe, columns=[score_column, label_column]
            )

        self.dataframe = dataframe
        self.label_column = label_column
        self.score_column = score_column
        self.date_column = date_column
        self.residuals = dataframe[label_column] - dataframe[score_column]
        self.decimals = decimals
        self.result = None

    def __str__(self):
        return self.__class__.__name__

    def kolmogorov_smirnov_test(
        self,
        dist="norm",
        pvalmethod="table",
        result_wrapper_key: str = "normality",
        flatten: bool = True,
        include_over_time: bool = True
    ) -> dict:
        """
        Test assumes normal or exponential distribution using Lilliefors' test. Lilliefors' test is
        a Kolmogorov-Smirnov test with estimated parameters.

        Args:
            dist ({'norm', 'exp'}, optional): The assumed distribution. Defaults to "norm".

            pvalmethod ({'approx', 'table'}, optional): The method used to compute the p-value of
                the test statistic. In general, 'table' is preferred and makes use of a very large
                simulation. 'approx' is only valid for normality. If dist = 'exp', 'table' is
                always used. 'approx' uses the approximation formula of Dalal and Wilkinson, valid
                for p-values < 0.1. If p-value > 0.1, then the result of 'table' is returned.
                Defaults to "table".

            result_wrapper_key (str): Provides a key to wrap the result dict (flattened results are outside this entry).

            flatten (bool): Surfaces flat metrics to top level of return dictionary.

            include_over_time (bool): Includes a new section with metrics calculation over a prediction date column.

        Returns:
            dict: Test results, including:
                ks_statistic (float): Kolmogorov-Smirnov test statistic with estimated mean and
                    variance.
                ks_p_value (float): KS test p-value. If the p-value is lower than some threshold,
                    e.g. 0.05, then we can reject the Null hypothesis that the sample comes from
                    a normal distribution.

        Notes:
            Built on: statsmodels>=0.12.2,<=0.13.2
            https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.kstest_normal.html
        """
        values_over_time = {}
        test_results = {
            "test_name": "Kolmogorov-Smirnov",
            "test_category": "normality",
            "test_type": "kolmogorov_smirnov",
            "test_id": "normality_kolmogorov_smirnov",
            "values": self.__calculate_kolmogorov_smirnov(dist, pvalmethod),
        }

        if include_over_time and self.date_column is not None:
            values_over_time = diagnostics_over_time(
                self,
                self.__calculate_kolmogorov_smirnov,
                result_wrapper_key=result_wrapper_key,
                test_name="Kolmogorov-Smirnov",
                test_title="Normality",
                custom_args={
                    "dist": dist,
                    "pvalmethod": pvalmethod
                }
            )
        self.result = values_over_time if values_over_time else {}
        if flatten:
            self.result.update({
                # Top-level metrics
                "ks_statistic": test_results["values"]["ks_statistic"],
                "ks_p_value": test_results["values"]["ks_p_value"],
                # Vanilla NormalityMetrics output
                result_wrapper_key: [test_results],
            })
        else:
            self.result.update({result_wrapper_key: [test_results]})
        return self.result
    
    def __calculate_kolmogorov_smirnov(self, dist, pvalmethod):
        """
        Test assumes normal or exponential distribution using Lilliefors' test. Lilliefors' test is
        a Kolmogorov-Smirnov test with estimated parameters.
        """
        values = kstest_normal(
            # x: array_like, 1d. Data to test.
            x=numpy.array(self.residuals),
            dist=dist,
            pvalmethod=pvalmethod,
        )

        values_dict = {
            "ks_statistic": numpy.round(values[0], self.decimals),
            "ks_p_value": numpy.round(values[1], self.decimals),
        }

        # Change numpy.nan to None if present
        return fix_numpy_nans_in_dict(dictionary=values_dict)

    def anderson_darling_test(
        self, axis=0, result_wrapper_key: str = "normality", flatten: bool = True, include_over_time = True
    ) -> dict:
        """
        Anderson-Darling test for normal distribution with unknown mean and variance.

        Args:
            axis (int, optional):  The axis to perform the test along. Defaults to 0.

            result_wrapper_key (str): Provides a key to wrap the result dict (flattened results are outside this entry).

            flatten (bool): Surfaces flat metrics to top level of return dictionary.

            include_over_time (bool): Includes a new section with metrics calculation over a prediction date column.

        Returns:
            dict: Test results, including:
                ad_statistic (float): The Anderson-Darling test statistic.
                ad_p_value (float): The p-value for hypothesis that the data comes from a normal
                    distribution with unknown mean and variance.

        Notes:
            Built on: statsmodels>=0.12.2,<=0.13.2
            https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.normal_ad.html
        """
        values_over_time = {}
        test_results = {
            "test_name": "Anderson-Darling",
            "test_category": "normality",
            "test_type": "anderson_darling",
            "test_id": "normality_anderson_darling",
            "values": self.__calculate_anderson_darling(axis),
        }

        if include_over_time and self.date_column is not None:
            values_over_time = diagnostics_over_time(
                self,
                self.__calculate_anderson_darling,
                result_wrapper_key=result_wrapper_key,
                test_name="Anderson-Darling",
                test_title="Normality",
                custom_args={
                    "axis": axis
                }
            )
        self.result = values_over_time if values_over_time else {}
        if flatten:
            self.result.update({
                # Top-level metrics
                "ad_statistic": test_results["values"]["ad_statistic"],
                "ad_p_value": test_results["values"]["ad_p_value"],
                # Vanilla NormalityMetrics output
                result_wrapper_key: [test_results],
            })
        else:
            self.result.update({result_wrapper_key: [test_results]})
        return self.result
    
    def __calculate_anderson_darling(self, axis):
        """
        Anderson-Darling test for normal distribution with unknown mean and variance.
        """
        values = normal_ad(
            # x: array_like. The data array.
            x=self.residuals,
            axis=axis,
        )

        values_dict = {
            "ad_statistic": numpy.round(values[0], self.decimals),
            "ad_p_value": numpy.round(values[1], self.decimals),
        }

        # Change numpy.nan to None if present
        return fix_numpy_nans_in_dict(dictionary=values_dict)

    def cramer_von_mises_test(
        self,
        cdf="norm",
        args=(),
        result_wrapper_key: str = "normality",
        flatten: bool = True,
        include_over_time: bool = True
    ) -> dict:
        """
        Performs the one-sample Cramer-von Mises test for goodness of fit. This performs a test of
        the goodness of fit of a cumulative distribution function (cdf) :math:`F` compared to the
        empirical distribution function :math:`F_n` of observed random variates :math:`X_1, ..., X_n`
        that are assumed to be independent and identically distributed. The null hypothesis is that
        the :math:`X_i` have cumulative distribution:math:`F`.

        Args:
            cdf ({str, callable}, optional): The cumulative distribution function :math:`F` to test
                the observations against. If a string, it should be the name of a distribution in
                `scipy.stats`. If a callable, that callable is used to calculate the cdf:
                ``cdf(x, *args) -> float``. Defaults to "norm".

            args (tuple, optional): Distribution parameters. These are assumed to be known.

            result_wrapper_key (str): Provides a key to wrap the result dict (flattened results are outside this entry).

            flatten (bool): Surfaces flat metrics to top level of return dictionary.

            include_over_time (bool): Includes a new section with metrics calculation over a prediction date column.
        Returns:
            dict: Test results, including:
                cvm_statistic (float): The Cramer-von Mises test statistic.
                cvm_p_value (float): The p-value of the Cramer-von Mises test.

        Notes:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cramervonmises.html
        """
        values_over_time = {}
        test_results = {
            "test_name": "Cramer-von Mises",
            "test_category": "normality",
            "test_type": "cramer_von_mises",
            "test_id": "normality_cramer_von_mises",
            "values": self.__calculate_cramer_von_mises(cdf, args),
        }

        if include_over_time and self.date_column is not None:
            values_over_time = diagnostics_over_time(
                self,
                self.__calculate_cramer_von_mises,
                result_wrapper_key=result_wrapper_key,
                test_name="Cramer-von Mises",
                test_title="Normality",
                custom_args={
                    "cdf": cdf,
                    "args": args
                }
            )
        self.result = values_over_time if values_over_time else {}
        if flatten:
            self.result.update({
                # Top-level metrics
                "cvm_statistic": test_results["values"]["cvm_statistic"],
                "cvm_p_value": test_results["values"]["cvm_p_value"],
                # Vanilla NormalityMetrics output
                result_wrapper_key: [test_results],
            })
        else:
            self.result.update({result_wrapper_key: [test_results]})
        return self.result
    
    def __calculate_cramer_von_mises(self, cdf, args):
        """
        Performs the one-sample Cramer-von Mises test for goodness of fit. This performs a test of
        the goodness of fit of a cumulative distribution function (cdf) :math:`F` compared to the
        empirical distribution function :math:`F_n` of observed random variates :math:`X_1, ..., X_n`
        that are assumed to be independent and identically distributed. The null hypothesis is that
        the :math:`X_i` have cumulative distribution:math:`F`.
        """
        values = cramervonmises(
            # rvs: array_like. A 1-D array of observed values of the random variables :math:`X_i`.
            rvs=self.residuals,
            cdf=cdf,
            args=args,
        )

        values_dict = {
            "cvm_statistic": numpy.round(values.statistic, self.decimals),
            "cvm_p_value": numpy.round(values.pvalue, self.decimals),
        }

        # Change numpy.nan to None if present
        return fix_numpy_nans_in_dict(dictionary=values_dict)


class AutocorrelationMetrics:
    """
    Class to comput autocorrelation metrics.

    Args:
        dataframe (pandas.DataFrame): Input DataFrame.

        job_json (dict): JSON dictionary with the metadata of the model.

        score_column (str): Model output.

        label_column (str): Ground truth.

        decimals (Optional[int], optional): Rounding parameter. Defaults to 4.

        date_column (str): Column containing dates for over time metrics.

    Examples:
        Load 'boston house prices' dataset and compute autocorrelation metrics:

        >>> import pandas
        >>> from pprint import pprint
        >>> from sklearn.datasets import load_boston

        >>> # load data
        >>> boston = load_boston()
        >>> X = pandas.DataFrame(boston.data, columns=boston.feature_names)
        >>> X.drop('CHAS', axis=1, inplace=True)
        >>> y = pandas.Series(boston.target, name='MEDV')

        >>> # Run linear regression
        >>> import statsmodels.api as sm
        >>> X_constant = sm.add_constant(X)
        >>> lin_reg = sm.OLS(y,X_constant).fit(random=0)

        >>> # Get predictions
        >>> X_constant["MEDV_pred"] = lin_reg.predict(X_constant).round(1)
        >>> # Add ground truth to DataFrame
        >>> X_constant["MEDV"] = y

        >>> from modelop.stats.diagnostics import AutocorrelationMetrics
        >>> autocorrelation_metrics = AutocorrelationMetrics(
        ...     dataframe=X_constant,
        ...     score_column="MEDV_pred",
        ...     label_column="MEDV",
        ... )

        >>> pprint(
        ...     autocorrelation_metrics.durbin_watson_test(
        ...         flatten=True
        ...     ),
        ...     sort_dicts=False
        ... )
        {'dw_statistic': 1.0172,
         'autocorrelation': [{'test_name': 'Durbin-Watson',
                              'test_category': 'autocorrelation',
                              'test_type': 'durbin_watson',
                              'test_id': 'autocorrelation_durbin_watson',
                              'values': {'dw_statistic': 1.0172}}]}
    """

    def __init__(
        self,
        dataframe: pandas.DataFrame,
        job_json: dict = None,
        score_column: str = None,
        label_column: str = None,
        decimals: Optional[int] = 4,
        date_column: str = None
    ) -> None:

        assert isinstance(
            dataframe, pandas.DataFrame
        ), "dataframe should be of type (pandas.DataFrame)."

        if job_json is not None:
            logger.info(
                "Parameter 'job_json' is present and will be used to extract "
                "'score_column', and 'label_column'."
            )
            input_schema_definition = infer.extract_input_schema(job_json)
            monitoring_parameters = infer.set_monitoring_parameters(
                schema_json=input_schema_definition, check_schema=True
            )
            score_column = monitoring_parameters["score_column"]
            label_column = monitoring_parameters["label_column"]
            date_column = monitoring_parameters["date_column"]
        else:
            if score_column is None or label_column is None:
                missing_args_error = (
                    "Parameter 'job_json' is not present, "
                    "but one of 'score_column' or 'label_column' was not provided. "
                    "'score_column' and 'label_column' input parameters are "
                    "required if 'job_json' is not provided."
                )
                logger.error(missing_args_error)
                raise Exception(missing_args_error)

        check_input_types(
            inputs=[{"score_column": score_column}, {"label_column": label_column}],
            types=(str),
        )

        if date_column is not None:
            check_input_types(inputs=[{"date_column": date_column}], types=(str))

        if date_column is not None:
            check_columns_in_dataframe(
                dataframe=dataframe, columns=[score_column, label_column, date_column]
            )
        else:
            check_columns_in_dataframe(
                dataframe=dataframe, columns=[score_column, label_column]
            )

        self.dataframe = dataframe
        self.label_column = label_column
        self.score_column = score_column
        self.date_column = date_column
        self.residuals = dataframe[label_column] - dataframe[score_column]
        self.decimals = decimals
        self.result = None

    def __str__(self):
        return self.__class__.__name__

    def durbin_watson_test(
        self,
        axis: int = 0,
        result_wrapper_key: str = "autocorrelation",
        flatten: bool = True,
        include_over_time: bool = True
    ) -> dict:
        """
        Calculates the Durbin-Watson statistic.

        Args:
            axis (int, optional): Axis to use if data has more than 1 dimension. Defaults to 0.

            result_wrapper_key (str): Provides a key to wrap the result dict.

            flatten (bool): Surfaces flat metrics to top level of return dictionary.

            include_over_time (bool): Includes a new section with metrics calculation over a prediction date column.

        Returns:
            dict: Test results, including:
                dw_statistic (float): The Durbin-Watson statistic.

        Notes:
            Built on: statsmodels>=0.12.2,<=0.13.2
            https://www.statsmodels.org/stable/generated/statsmodels.stats.stattools.durbin_watson.html

            The null hypothesis is that there is no serial correlation in the residuals.
            The test statistic is approximately equal to 2*(1-r) where r is the sample
            autocorrelation of the residuals. Thus, for r == 0, indicating no serial correlation,
            the test statistic equals 2. This statistic will always be between 0 and 4. The closer
            to 0 the statistic, the more evidence for positive serial correlation. The closer to 4,
            the more evidence for negative serial correlation.
        """

        values_over_time = {}
        test_results = {
            "test_name": "Durbin-Watson",
            "test_category": "autocorrelation",
            "test_type": "durbin_watson",
            "test_id": "autocorrelation_durbin_watson",
            "values": self.__calculate_durbin_watson(axis),
        }

        if include_over_time and self.date_column is not None:
            values_over_time = diagnostics_over_time(
                self,
                self.__calculate_durbin_watson,
                result_wrapper_key=result_wrapper_key,
                test_name="Durbin-Watson",
                test_title="Autocorrelation",
                custom_args={
                    "axis": axis
                }
            )
        self.result = values_over_time if values_over_time else {}
        if flatten:
            self.result.update({
                # Top-level metrics
                "dw_statistic": test_results["values"]["dw_statistic"],
                # Vanilla AutocorrelationMetrics output
                result_wrapper_key: [test_results],
            })
        else:
            self.result.update({result_wrapper_key: [test_results]})
        return self.result

    def __calculate_durbin_watson(self, axis):
        """
        Calculates the Durbin-Watson statistic.
        """
        values = durbin_watson(
            # resids: array_like. Regression model residuals.
            resids=self.residuals,
            axis=axis,
        )

        values_dict = {"dw_statistic": numpy.round(values, self.decimals)}

        # Change numpy.nan to None if present
        return fix_numpy_nans_in_dict(dictionary=values_dict)

class LinearityMetrics:
    """
    Class to compute linearity metrics.

    Args:
        dataframe (pandas.DataFrame): Input DataFrame.

        job_json (dict): JSON dictionary with the metadata of the model.

        label_column (str): Ground truth.

        numerical_predictors (Optional[list], optional): Predictor variables. Defaults to None.

        decimals (Optional[int], optional): Rounding parameter. Defaults to 4.

        date_column (str): Column containing dates for over time metrics.

    Examples:
        Load 'boston house prices' dataset and compute Pearson Correlation metrics:

        >>> import pandas
        >>> from pprint import pprint
        >>> from sklearn.datasets import load_boston

        >>> # load data
        >>> boston = load_boston()
        >>> X = pandas.DataFrame(boston.data, columns=boston.feature_names)
        >>> X.drop('CHAS', axis=1, inplace=True)
        >>> y = pandas.Series(boston.target, name='MEDV')
        >>> X["MEDV"] = y

        >>> from modelop.stats.diagnostics import LinearityMetrics
        >>> linearity_metrics = LinearityMetrics(
        ...     dataframe=X,
        ...     label_column="MEDV",
        ...     numerical_predictors=[
        ...         'CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
        ...         'PTRATIO', 'B', 'LSTAT'
        ...     ]
        ... )

        >>> pprint(
        ...     linearity_metrics.pearson_correlation(
        ...         include_min_max_features=True
        ...     ),
        ...     sort_dicts=False
        ... )
        {'Linearity_minPearsonCorrelationValue': -0.7377,
         'Linearity_minPearsonCorrelationValueFeature': 'LSTAT',
         'Linearity_maxPearsonCorrelationValue': 0.6954,
         'Linearity_maxPearsonCorrelationValueFeature': 'RM',
         'linearity': [{'test_name': 'Pearson Correlation',
                        'test_category': 'linearity',
                        'test_type': 'pearson_correlation',
                        'test_id': 'linearity_pearson_correlation',
                        'metric': 'correlation_to_MEDV',
                        'values': {'MEDV': 1,
                                   'const': None,
                                   'CRIM': -0.3883,
                                   'ZN': 0.3604,
                                   'INDUS': -0.4837,
                                   'NOX': -0.4273,
                                   'RM': 0.6954,
                                   'AGE': -0.377,
                                   'DIS': 0.2499,
                                   'RAD': -0.3816,
                                   'TAX': -0.4685,
                                   'PTRATIO': -0.5078,
                                   'B': 0.3335,
                                   'LSTAT': -0.7377}}]}

        The example below computes Box-Tidwell p-values (p-values of log interaction terms)

        >>> X = pandas.DataFrame(boston.data, columns=boston.feature_names)
        >>> X.drop('CHAS', axis=1, inplace=True)
        >>> y = pandas.Series(boston.target, name='MEDV')
        >>> X["MEDV_bool"] = y<23

        >>> from modelop.stats.diagnostics import LinearityMetrics
        >>> linearity_metrics = LinearityMetrics(
        ...     dataframe=X,
        ...     label_column="MEDV_bool",
        ...     numerical_predictors=[
        ...         'CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
        ...         'PTRATIO', 'B', 'LSTAT'
        ...     ]
        ... )

        >>> pprint(
        ...     linearity_metrics.box_tidwell(
        ...         include_min_max_features=True
        ...     ),
        ...     sort_dicts=False
        ... )
        {'Linearity_minBoxTidwellPValueValue': 0.012,
         'Linearity_minBoxTidwellPValueValueFeature': 'RM_log_int',
         'Linearity_maxBoxTidwellPValueValue': 0.6709,
         'Linearity_maxBoxTidwellPValueValueFeature': 'RAD_log_int',
         'linearity': [{'test_name': 'Box-Tidwell',
                        'test_category': 'linearity',
                        'test_type': 'box_tidwell',
                        'test_id': 'linearity_box_tidwell',
                        'metric': 'p_value',
                        'values': {'CRIM_log_int': 0.3802,
                                   'ZN_log_int': 0.5932,
                                   'INDUS_log_int': 0.1544,
                                   'NOX_log_int': 0.0942,
                                   'RM_log_int': 0.012,
                                   'AGE_log_int': 0.1126,
                                   'DIS_log_int': 0.1401,
                                   'RAD_log_int': 0.6709,
                                   'TAX_log_int': 0.2149,
                                   'PTRATIO_log_int': 0.1288,
                                   'B_log_int': 0.0951,
                                   'LSTAT_log_int': 0.0263}}]}
    """

    def __init__(
        self,
        dataframe: pandas.DataFrame,
        job_json: dict = None,
        label_column: str = None,
        numerical_predictors: Optional[list] = None,
        decimals: Optional[int] = 4,
        date_column: str = None
    ) -> None:

        assert isinstance(
            dataframe, pandas.DataFrame
        ), "dataframe should be of type (pandas.DataFrame)."

        if job_json is not None:
            logger.info(
                "Parameter 'job_json' is present and will be used to extract "
                "'label_column' and 'numerical_predictors'."
            )
            input_schema_definition = infer.extract_input_schema(job_json)
            monitoring_parameters = infer.set_monitoring_parameters(
                schema_json=input_schema_definition, check_schema=True
            )
            label_column = monitoring_parameters["label_column"]
            numerical_predictors = monitoring_parameters["numerical_columns"]
            date_column = monitoring_parameters["date_column"]
        else:
            if label_column is None:
                missing_args_error = (
                    "Parameter 'job_json' is not present, "
                    "but 'label_column' was not provided. "
                    "'label_column' input parameter is "
                    "required if 'job_json' is not provided."
                )
                logger.error(missing_args_error)
                raise Exception(missing_args_error)

        check_input_types(
            inputs=[
                {"numerical_predictors": numerical_predictors},
            ],
            types=(list, type(None)),
        )

        check_input_types(
            inputs=[{"label_column": label_column}],
            types=(str),
        )

        if date_column is not None:
            check_input_types(inputs=[{"date_column": date_column}], types=(str))

        if date_column is not None:
            check_columns_in_dataframe(
                dataframe=dataframe, columns=[label_column, date_column]
            )
        else:
            check_columns_in_dataframe(
                dataframe=dataframe, columns=[label_column]
            )

        if numerical_predictors is not None:
            check_columns_in_dataframe(
                dataframe=dataframe, columns=numerical_predictors
            )

        # Infer numerical predictors if not specified
        else:
            numerical_predictors = [
                col
                for col in dataframe.columns
                if (
                    numpy.isin(
                        dataframe.dtypes[col],
                        ["float32", "float64", "int32", "int64", "uint8"],
                    )
                    and not numpy.isin(col, [label_column])
                )
            ]
            print(
                "Identified numerical predictor(s): ", numerical_predictors
            ), "dataframe must contain at least 1 numerical_predictor column!"

        # Restricting dataframe to columns of interest
        if date_column is not None:
            dataframe = dataframe[numerical_predictors + [label_column, date_column]]
        else:
            dataframe = dataframe[numerical_predictors + [label_column]]

        # Run reordering of columns - add constant column if none provided
        self.dataframe = handle_constant_columns(
            dataframe,
        )
        self.numerical_predictors = list(self.dataframe.columns)
        self.numerical_predictors.remove(label_column)
        if date_column is not None and date_column in self.numerical_predictors:
            self.numerical_predictors.remove(date_column)
        self.label_column = label_column
        self.date_column = date_column
        self.decimals = decimals
        self.result = None

    def __str__(self):
        return self.__class__.__name__

    def pearson_correlation(
        self,
        result_wrapper_key: Optional[str] = "linearity",
        include_min_max_features: bool = True,
        include_over_time: bool = True
    ) -> dict:
        """
        Pearson correlation coefficient and p-value for testing non-correlation.

        Args:
            result_wrapper_key (str): Provides a key to wrap the result dict.

            include_min_max_features (bool): Provides the min and max values of the flattened results,
                and their corresponding feature names.

        include_over_time (bool): Includes a new section with metrics calculation over a prediction date column.

        Returns:
            dict: Test results, including
                Pearson correlation coefficient (float) for each numerical feature Vs label column
        """

        values_over_time = {}
        fixed_values = self.__calculate_pearson_correlation()
        test_results = {
            "test_name": "Pearson Correlation",
            "test_category": "linearity",
            "test_type": "pearson_correlation",
            "test_id": "linearity_pearson_correlation",
            "metric": "correlation_to_{}".format(self.label_column),
            "values": fixed_values,
        }

        self.result = {}

        if include_over_time and self.date_column is not None:
            values_over_time = diagnostics_over_time(
                self,
                self.__calculate_pearson_correlation,
                result_wrapper_key=result_wrapper_key,
                test_name="Pearson Correlation",
                test_title="Linearity",
                custom_args={}
            )

        self.result.update(values_over_time)

        if include_min_max_features:
            correlation_dict_no_label = {
                k: v for k, v in fixed_values.items() if k != self.label_column
            }
            min_max_dict = get_min_max_values_keys_from_dict(
                values_dict=correlation_dict_no_label
            )

            self.result["Linearity_minPearsonCorrelationValue"] = min_max_dict[
                "min_value"
            ]
            self.result["Linearity_minPearsonCorrelationValueFeature"] = min_max_dict[
                "min_feature"
            ]
            self.result["Linearity_maxPearsonCorrelationValue"] = min_max_dict[
                "max_value"
            ]
            self.result["Linearity_maxPearsonCorrelationValueFeature"] = min_max_dict[
                "max_feature"
            ]

        # Vanilla LinearityMetrics output
        self.result[result_wrapper_key] = [test_results]
        return self.result
    
    def __calculate_pearson_correlation(self):
        """
        Pearson correlation coefficient and p-value for testing non-correlation.
        """
        values = {self.label_column: 1}
        for feature in self.numerical_predictors:
            try:
                values[feature] = numpy.round(
                    pearsonr(
                        x=self.dataframe[feature], y=self.dataframe[self.label_column]
                    )[0],
                    self.decimals,
                )
                if ma.is_masked(values[feature]):
                    values[feature] = values[feature].data.__float__()
            except ConstantInputWarning:
                logger.exception(
                    "Constant values encountered in column %s! Pearson Coefficient is undefined. Setting it to NULL.",
                    feature,
                )
                values[feature] = None

        # Change numpy.nan to None if present
        return fix_numpy_nans_in_dict(dictionary=values)

    def box_tidwell(
        self,
        result_wrapper_key: Optional[str] = "linearity",
        include_min_max_features: bool = True,
        include_over_time: bool = True
    ) -> dict:
        """
        Box-Tidwell Test, for exogenous variables.

        The Box-Tidwell test is used to check for linearity between the predictors and the logit.
        This is done by adding log-transformed interaction terms between the continuous independent
        variables and their corresponding natural log into the model.

        If you have more than one continuous variable, you should include the same number of
        interaction terms in the model. With the interaction terms included, we can re-run
        the logistic regression and review the results.

        What we need to do is check the statistical significance of the interaction terms
        based on their p-values.

        If an interaction term has a large p-value, this implies that the independent variable is
        linearly related to the logit of the outcome variable and that the assumption is satisfied.

        On the contrary, if the term is statistically significant (i.e., small p-value), this indicates the presence
        of non-linearity between the variable and the logit.

        One solution is to perform transformations by incorporating higher-order polynomial terms to capture
        the non-linearity.

        Args:
            result_wrapper_key (str): Provides a key to wrap the result dict.

            include_min_max_features (bool): Provides the min and max values of the flattened results,
                and their corresponding feature names.

            include_over_time (bool): Includes a new section with metrics calculation over a prediction date column.

        Returns:
            dict: Test results, including
                P-value for each x_log_int, the logarithmic interaction term for x*log(x).

        Notes:
            This function does not save the auxiliary regression.

        References:
            https://towardsdatascience.com/assumptions-of-logistic-regression-clearly-explained-44d85a22b290#:~:text=The%20Box%2DTidwell%20test%20is,natural%20log%20into%20the%20model.
        """

        values_over_time = {}
        fixed_values = self.__calculate_box_tidwell()
        test_results = {
            "test_name": "Box-Tidwell",
            "test_category": "linearity",
            "test_type": "box_tidwell",
            "test_id": "linearity_box_tidwell",
            "metric": "p_value",
            "values": fixed_values,
        }

        if include_over_time and self.date_column is not None:
            values_over_time = diagnostics_over_time(
                self,
                self.__calculate_box_tidwell,
                result_wrapper_key=result_wrapper_key,
                test_name="Box-Tidwell",
                test_title="Linearity",
                custom_args={}
            )

        self.result = values_over_time if values_over_time else {}

        if include_min_max_features:
            correlation_dict_no_label = {
                k: v for k, v in fixed_values.items() if k != self.label_column
            }
            min_max_dict = get_min_max_values_keys_from_dict(
                values_dict=correlation_dict_no_label
            )

            self.result["Linearity_minBoxTidwellPValueValue"] = min_max_dict[
                "min_value"
            ]
            self.result["Linearity_minBoxTidwellPValueValueFeature"] = min_max_dict[
                "min_feature"
            ]
            self.result["Linearity_maxBoxTidwellPValueValue"] = min_max_dict[
                "max_value"
            ]
            self.result["Linearity_maxBoxTidwellPValueValueFeature"] = min_max_dict[
                "max_feature"
            ]

        self.result[result_wrapper_key] = [test_results]
        return self.result
    
    def __calculate_box_tidwell(self):
        """
        Box-Tidwell Test, for exogenous variables.

        The Box-Tidwell test is used to check for linearity between the predictors and the logit.
        This is done by adding log-transformed interaction terms between the continuous independent
        variables and their corresponding natural log into the model.

        If you have more than one continuous variable, you should include the same number of
        interaction terms in the model. With the interaction terms included, we can re-run
        the logistic regression and review the results.

        What we need to do is check the statistical significance of the interaction terms
        based on their p-values.

        If an interaction term has a large p-value, this implies that the independent variable is
        linearly related to the logit of the outcome variable and that the assumption is satisfied.

        On the contrary, if the term is statistically significant (i.e., small p-value), this indicates the presence
        of non-linearity between the variable and the logit.

        One solution is to perform transformations by incorporating higher-order polynomial terms to capture
        the non-linearity.
        """
        # Define continuous variables
        # Redefine independent variables to include interaction terms

        interactions_df = pandas.DataFrame()

        for var in self.numerical_predictors:
            if self.dataframe[var].nunique() > 1:
                var_min = numpy.min(self.dataframe[var])
                if var_min < 1:
                    interactions_df[f"{var}_log_int"] = self.dataframe[var].apply(
                        lambda x: (x - var_min + 1) * numpy.log(x - var_min + 1)
                    )  # np.log = natural log
                else:
                    interactions_df[f"{var}_log_int"] = self.dataframe[var].apply(
                        lambda x: (x) * numpy.log(x)
                    )

        # Build model and fit the data (using statsmodel's Logit)
        logit_results = Logit(
            self.dataframe[self.label_column],
            pandas.concat(
                [self.dataframe[self.numerical_predictors], interactions_df], axis=1
            ),
        ).fit(disp=0) # disp=0 suppresses convergence info

        # p-values of log interaction terms start at index int((len(logit_results.pvalues) - 1) / 2) + 1
        values = numpy.round(
            logit_results.pvalues[int((len(logit_results.pvalues) - 1) / 2) + 1 :],
            self.decimals,
        ).to_dict()

        return fix_numpy_nans_in_dict(dictionary=values)


class MulticollinearityMetrics:
    """
    Class to compute multicollinearity metrics.

    Args:
        dataframe (pandas.DataFrame): Input DataFrame.

        job_json (dict): JSON dictionary with the metadata of the model.

        numerical_predictors (List): Predictor variables.

        decimals (Optional[int], optional): Rounding parameter. Defaults to 4.

        date_column (str): Column containing dates for over time metrics.

    Examples:
        Load 'boston house prices' dataset and compute multicollinearity metrics:

        >>> import pandas
        >>> from pprint import pprint
        >>> from sklearn.datasets import load_boston

        >>> # load data
        >>> boston = load_boston()
        >>> X = pandas.DataFrame(boston.data, columns=boston.feature_names)
        >>> X.drop('CHAS', axis=1, inplace=True)

        >>> from modelop.stats.diagnostics import MulticollinearityMetrics
        >>> multicollinearity_metrics = MulticollinearityMetrics(
        ...     dataframe=X,
        ...     numerical_predictors=[
        ...         'CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
        ...         'PTRATIO', 'B', 'LSTAT'
        ...     ]
        ... )

        >>> pprint(
        ... multicollinearity_metrics.variance_inflation_factor(
        ...     flatten=True,
        ...     include_min_max_features=True,
        ...     ),
        ...     sort_dicts=False
        ... )
        {'CRIM_vif': 2.0955,
         'ZN_vif': 2.8433,
         'INDUS_vif': 14.336,
         'NOX_vif': 73.6483,
         'RM_vif': 77.5485,
         'AGE_vif': 21.3362,
         'DIS_vif': 14.699,
         'RAD_vif': 15.003,
         'TAX_vif': 60.3605,
         'PTRATIO_vif': 84.2342,
         'B_vif': 20.0278,
         'LSTAT_vif': 11.0699,
         'Multicollinearity_minVIFValue': 2.0955,
         'Multicollinearity_minVIFValueFeature': 'CRIM',
         'Multicollinearity_maxVIFValue': 84.2342,
         'Multicollinearity_maxVIFValueFeature': 'PTRATIO',
         'multicollinearity': [{'test_name': 'Variance Inflation Factor',
                                'test_category': 'multicollinearity',
                                'test_type': 'variance_inflation_factor',
                                'test_id': 'multicollinearity_variance_inflation_factor',
                                'values': {'CRIM': 2.0955,
                                           'ZN': 2.8433,
                                           'INDUS': 14.336,
                                           'NOX': 73.6483,
                                           'RM': 77.5485,
                                           'AGE': 21.3362,
                                           'DIS': 14.699,
                                           'RAD': 15.003,
                                           'TAX': 60.3605,
                                           'PTRATIO': 84.2342,
                                           'B': 20.0278,
                                           'LSTAT': 11.0699}}]}
    """

    def __init__(
        self,
        dataframe: pandas.DataFrame,
        job_json: dict = None,
        numerical_predictors: Optional[list] = None,
        decimals: Optional[int] = 4,
        date_column: str = None,
    ) -> None:

        assert isinstance(
            dataframe, pandas.DataFrame
        ), "dataframe should be of type (pandas.DataFrame)."

        if job_json is not None:
            logger.info(
                "Parameter 'job_json' is present and will be used to extract "
                "'numerical_predictors'."
            )
            input_schema_definition = infer.extract_input_schema(job_json)
            monitoring_parameters = infer.set_monitoring_parameters(
                schema_json=input_schema_definition, check_schema=True
            )
            numerical_predictors = monitoring_parameters["numerical_columns"]
            date_column = monitoring_parameters["date_column"]

        check_input_types(
            inputs=[
                {"numerical_predictors": numerical_predictors},
            ],
            types=(list, type(None)),
        )

        if date_column is not None:
            check_input_types(inputs=[{"date_column": date_column}], types=(str))

        check_columns_in_dataframe(dataframe=dataframe, columns=numerical_predictors)

        if date_column is not None:
            check_columns_in_dataframe(dataframe=dataframe, columns=[date_column])
        
        self.numerical_predictors = numerical_predictors
        if date_column is not None and date_column in self.numerical_predictors:
            self.numerical_predictors.remove(date_column)
        # Cast all numerical columns as floats - required to avoid OLS.fit errors
        if date_column is not None:
            dataframe[numerical_predictors] = dataframe[numerical_predictors].astype(float)
            self.dataframe = dataframe[numerical_predictors + [date_column]]
        else:
            self.dataframe = dataframe[numerical_predictors].astype(float)
        self.date_column = date_column
        self.decimals = decimals
        self.result = None

    def __str__(self):
        return self.__class__.__name__

    def variance_inflation_factor(
        self,
        flatten: bool = True,
        result_wrapper_key: str = "multicollinearity",
        include_min_max_features: bool = True,
        include_over_time: bool = True
    ):
        """
        Computes VIF (Variance Inflation Factor) metrics on a scored & labeled dataset.

        Args:
            result_wrapper_key (str): Provides a key to wrap the result dict.

            include_min_max_features (bool): Provides the min and max values of the flattened results,
                and their corresponding feature names.

            flatten (bool): Surfaces flat metrics to top level of return dictionary.

            include_over_time (bool): Includes a new section with metrics calculation over a prediction date column.

        Returns:
            VIF metrics.
        """

        values_over_time = {}
        test_results = {
            "test_name": "Variance Inflation Factor",
            "test_category": "multicollinearity",
            "test_type": "variance_inflation_factor",
            "test_id": "multicollinearity_variance_inflation_factor",
            "values": self.__calculate_vairance_inflation_factor(),
        }

        if include_over_time and self.date_column is not None:
            values_over_time = diagnostics_over_time(
                self,
                self.__calculate_vairance_inflation_factor,
                result_wrapper_key=result_wrapper_key,
                test_name="Variance Inflation Factor",
                test_title="Multicollinearity",
                custom_args={}
            )

        self.result = values_over_time if values_over_time else {}

        if flatten:
            self.result.update({
                # Top-level metrics
                str(feature + "_vif"): test_results["values"][feature]
                for feature in test_results["values"].keys()
            })

        if include_min_max_features:
            min_max_dict = get_min_max_values_keys_from_dict(
                values_dict=test_results["values"]
            )

            self.result["Multicollinearity_minVIFValue"] = min_max_dict["min_value"]
            self.result["Multicollinearity_minVIFValueFeature"] = min_max_dict[
                "min_feature"
            ]
            self.result["Multicollinearity_maxVIFValue"] = min_max_dict["max_value"]
            self.result["Multicollinearity_maxVIFValueFeature"] = min_max_dict[
                "max_feature"
            ]

        self.result[result_wrapper_key] = [test_results]
        return self.result
    
    def __calculate_vairance_inflation_factor(self):
        """
        Computes VIF (Variance Inflation Factor) metrics on a scored & labeled dataset.
        """
        # Create copy of dataframe, check for and drop nulls
        if isinstance(self.dataframe, pandas.Series):
            self.dataframe = self.dataframe.to_frame()
        if self.date_column and self.date_column in self.dataframe:
            dataframe = self.dataframe.drop(columns=self.date_column)
        else:
            dataframe = self.dataframe
        dataframe_copy_no_nulls = self.__check_and_drop_nulls(dataframe)
        values = vif(
            exog_df=dataframe_copy_no_nulls,
            columns=self.numerical_predictors,
            decimals=self.decimals
        )

        # Change numpy.nan to None if present
        return fix_numpy_nans_in_dict(dictionary=values)

    @staticmethod
    def __check_and_drop_nulls(
        dataframe: pandas.DataFrame,
    ) -> pandas.DataFrame:
        """
        A function to check for and drop NULLs from dataframe, and log actions.

        Args:
            dataframe (pandas.DataFrame): Pandas DataFrame.

        Returns:
            pandas.DataFrame: Copy of self.dataframe with dropped nulls.

        """
        dataframe_copy = dataframe.copy()

        # Checking for NULLs
        for column in dataframe.columns:
            column_null_count = dataframe_copy[column].isna().sum()
            if column_null_count > 0:
                logger.warning(
                    "Encountered %i NULL(s) in column '%s.' Instances of NULL(s) will be dropped prior to metrics computation.",
                    column_null_count,
                    column,
                )
                dataframe_copy = dataframe_copy.dropna(subset=[column])
        return dataframe_copy


def vif(exog_df: pandas.DataFrame, columns: List[str], decimals: int) -> dict:
    """
    Variance inflation factor, VIF, for one exogenous variable.

    The variance inflation factor is a measure for the increase of the
    variance of the parameter estimates if an additional variable, given by
    exog_idx is added to the linear regression. It is a measure for
    multicollinearity of the design matrix, exog_df.

    One recommendation is that if VIF is greater than 5, then the explanatory
    variable given by exog_idx is highly collinear with the other explanatory
    variables, and the parameter estimates will have large standard errors
    because of this.

    Args:
        exog_df (pandas.DataFrame): Design matrix with all explanatory
        variables, as for example used in regression
        columns (List(str)): List of exogenous variables in the columns of exog_df
        decimals (int): Number of decimals to round metrics to.

    Returns:
        dict: Variance inflation factor for each exogenous variable

    Notes:
        This function does not save the auxiliary regression.

    References:
        https://en.wikipedia.org/wiki/Variance_inflation_factor
    """

    k_vars = exog_df.shape[1]
    exog_array = numpy.asarray(exog_df)

    result = {}
    for feature in columns:
        exog_idx = list(exog_df.columns).index(feature)
        x_i = exog_array[:, exog_idx]
        mask = numpy.arange(k_vars) != exog_idx
        x_noti = exog_array[:, mask]
        r_squared_i = OLS(x_i, x_noti).fit().rsquared
        if r_squared_i != 1:
            v_i_f = 1.0 / (1.0 - r_squared_i)
            result[feature] = numpy.round(v_i_f, decimals)
        else:
            logger.warning(
                "VIF = Infinity encountered on feature %s! Setting value to None.",
                feature,
            )
            result[feature] = None

    return result


def handle_constant_columns(
    dataframe: pandas.DataFrame,
) -> pandas.DataFrame:
    """
    A function to check for a constant column in a dataframe and, if found, set it as the first column.
    In the case of no constant columns, add one.
    In the case of multiple constant columns, sets the first column found as the default constant column.

    Args:
        dataframe (pandas.DataFrame): Pandas DataFrame.

    Returns:
        (pandas.DataFrame): Pandas DataFrame with constant column placed first.
    """

    # Iterate through columns to check if column is constant
    constant_column_name = None
    for col in dataframe.columns:
        if dataframe[col].nunique() == 1:
            logger.info("Constant column found: %s.", col)
            constant_column_name = col
            break

    if not constant_column_name:
        logger.info(
            "No constant columns were found. One, named 'const', will be added."
        )
        constant_column_name = "const"
        dataframe.loc[:, constant_column_name] = 1

    dataframe = dataframe[
        [constant_column_name]
        + [col for col in dataframe.columns if col != constant_column_name]
    ]
    logger.info(
        "Constant column %s was placed first in dataframe.", constant_column_name
    )

    return dataframe

def diagnostics_over_time(    
    self,
    evaluation_function,
    result_wrapper_key: str = 'normality',
    test_name: str = '',
    test_title: str = '',
    custom_args: dict = {}
):
    """
    A function that computes the metrics as given by the evaluation_function, attempting to split the data by date
    :param self: The object holding the diagnostic data
    :param evaluation_function: The function to run per data split (by date)
    :param result_wrapper_key: The key used for the wrapping the whole test results. This will create a similar key
    :param test_name: The name of the specific test actually run.
    :param custom_args: A dict of additional kwargs to pass to the evaluation_function
    :return: A dictionary with a graph structure over time and the traditional results
    """
    if self.date_column is not None and self.date_column in self.dataframe:
        self.dataframe = self.dataframe.set_index(check_date_column(self.dataframe, self.date_column).dt.date)
        self.dataframe[self.date_column] = check_date_column(self.dataframe, self.date_column).dt.date
        self.dataframe = self.dataframe.sort_index()

        unique_dates = self.dataframe[self.date_column].unique()

        data = {}
        full_dataframe = self.dataframe
        if hasattr(self, "residuals"):
            original_residual = self.residuals
        if hasattr(self, "exog"):
            original_exog = self.exog
        for date in unique_dates:
            self.dataframe = full_dataframe
            self.dataframe = self.dataframe.loc[date]
            if hasattr(self, "residuals"):
                self.residuals = self.dataframe[self.label_column] - self.dataframe[self.score_column]
                if isinstance(self.residuals, int) or isinstance(self.residuals, float):
                    self.residuals = pandas.Series([self.residuals])
            if hasattr(self, "exog"):
                self.exog = self.dataframe[self.numerical_predictors]
            try:
                dated_values = evaluation_function(**custom_args)
            except Exception as err:
                if (hasattr(self, "residuals") and len(self.residuals) == 1) or (hasattr(self, "exog") and len(self.exog.shape) == 1) or isinstance(self.dataframe, pandas.Series):
                    logger.info(f"Only one record found for day={date}. Metrics cannot be calculated on only one data point. Skipping metrics calculation for day={date}.")
                else:
                    logger.error(str(err) + f". Skipping metrics calculation for day={date}")
                dated_values = None
            str_date = str(date)
            if dated_values:
                for metric in dated_values:
                    if metric not in data:
                        data[metric] = []
                    data[metric].append([str_date, dated_values[metric]])
        self.dataframe = full_dataframe
        if hasattr(self, "residuals"):
            self.residuals = original_residual
        if hasattr(self, "exog"):
            self.exog = original_exog
        over_time_key = result_wrapper_key + "_over_time"
        values_over_time = {over_time_key: {
            "title": (test_title if test_title else "Test") + " Over Time" + (" - " + test_name if test_name else ""),
            "x_axis_label": "Day",
            "y_axis_label": "Metric",
            "data": data
        },
            "firstPredictionDate": str(unique_dates.min()),
            "lastPredictionDate": str(unique_dates.max()),
        }
        return values_over_time


if __name__ == "__main__":
    print(doctest.testmod())
    print()
