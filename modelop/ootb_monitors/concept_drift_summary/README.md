# Concept Drift Monitor: Summary
This ModelOp Center monitor computes and compares Pandas summary on **output** data.

## Input Assets

| Type          | Number | Description                                           |
| ------------- | ------ | ----------------------------------------------------- |
| Baseline Data | **1**  | A dataset corresponding to training/reference data    |
| Comparator Data   | **1**  | A dataset corresponding to a slice of production data |

## Assumptions & Requirements
 - Underlying `BUSINESS_MODEL` being monitored has a **job json** asset.
 - Input data must contain:
     - 1 column with **role=score** (model output) 

## Execution
1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
2. `metrics` function instantiates the **Concept Drift Monitor** class and uses the job json asset to set the `target_column` (score column) and `output_type` (`numerical` vs. `categorical`).
3. The **Pandas.describe()** concept drift test is run.
   - If `output_type` is `numerical`, comparison between baseline and comparator score columns is done on `count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, and `max` values.
   - If `output_type` is `categorical`, comparison between baseline and comparator score columns is done on `count`, `unique`, `top`, and `freq` values.
4. Test result is returned under the list of `concept_drift` tests.

## Monitor Output

```JSON
{
    "concept_drift": [
        {
            "test_name": "Summary",
            "test_category": "concept_drift",
            "test_type": "summary",
            "metric": "pandas_describe",
            "test_id": "concept_drift_summary_pandas_describe",
            "values": {
                "numerical_comparisons": {
                    <target_column>: <comparison of count/mean/std/quantiles>
                },
                "categorical_comparisons": {
                    <target_column>: <comparison of count/unique/top/freq>
                }
            }
        }
    ]
}
```

> Note: Only one of `numerical_comparisons` or `categorical_comparisons` will be populated, depending on `output_type`.