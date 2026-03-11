# Data Drift Monitor: Comprehensive Analysis
This ModelOp Center monitor runs and compares **Kolmogorov-Smirnov**, **Epps-Singleton**, **Jensen-Shannon**, **Kullback-Leibler**, and **Pandas summary** on **input** data.

## Input Assets

| Type          | Number | Description                                           |
| ------------- | ------ | ----------------------------------------------------- |
| Baseline Data | **1**  | A dataset corresponding to training/reference data    |
| Sample Data   | **1**  | A dataset corresponding to a slice of production data |

## Assumptions & Requirements
 - Underlying `BUSINESS_MODEL` being monitored has a **job json** asset.
 - Input data contains at least one `numerical` column or one `categorical` column.

## Execution
1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
2. `metrics` function instantiates the **Data Drift Monitor** class and uses the job json asset to determine the `numerical_columns` and/or `categorical_columns` accordingly.
3. The **Epps-Singleton**, **Jensen-Shannon**, **Kullback-Leibler**, **Kolmogorov-Smirnov**, and **Pandas.describe()** data drift tests are run.
4. Each test result is appended to the list of `data_drift` tests to be returned by the model.

## Monitor Output

```JSON
{
    "data_drift": [
        <epps_singleton_test_result>,
        <jensen_shannon_test_result>,
        <kullback_leibler_test_result>,
        <kolmogorov_smirnov_test_result>,
        <pandas_summary_test_result>
    ]
}
```