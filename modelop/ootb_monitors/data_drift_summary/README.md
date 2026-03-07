# Data Drift Monitor: Summary
This ModelOp Center monitor computes and compares Pandas summaries on **input** data.

## Input Assets

| Type          | Number | Description                                           |
| ------------- | ------ | ----------------------------------------------------- |
| Baseline Data | **1**  | A dataset corresponding to training/reference data    |
| Comparator Data   | **1**  | A dataset corresponding to a slice of production data |

## Assumptions & Requirements
 - Underlying `BUSINESS_MODEL` being monitored has an **extended input schema** asset.
 - Input data contains at least one `numerical` column or one `categorical` column.

## Execution
1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
2. `metrics` function instantiates the **Data Drift Monitor** class and uses the job json asset to determine the `numerical_columns` and/or `categorical_columns` accordingly.
3. The **Pandas.describe()** data drift test is run.
   - For `numerical` features, comparisons between baseline and comparator datasets are done on `count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, and `max` values.
   - For `categorical` features, comparisons between baseline and comparator datasets are done on `count`, `unique`, `top`, and `freq` values.
4. Test result is returned under the list of `data_drift` tests.

## Monitor Output

```JSON
{
    "concept_drift": [
        {
            "test_name": "Summary",
            "test_category": "data_drift",
            "test_type": "summary",
            "metric": "pandas_describe",
            "test_id": "data_drift_summary_pandas_describe",
            "values": {
                "numerical_comparisons": {
                    <numerical_feature_1>: <comparison of count/mean/std/quantiles>,
                    ...:...,
                    <numerical_feature_n>: <comparison of count/mean/std/quantiles>
                },
                "categorical_comparisons": {
                    <categorical_feature_1>: <comparison of count/unique/top/freq>,
                    ...:...,
                    <categorical_feature_n>: <comparison of count/unique/top/freq>
                }
            }
        }
    ]
}
```