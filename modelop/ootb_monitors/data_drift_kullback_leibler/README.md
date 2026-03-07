# Data Drift Monitor: Kullback-Leibler
This ModelOp Center monitor computes **Kullback-Leibler divergences** on **input** data.

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
3. The **Kullback-Leibler** data drift test is run.
   - For `numerical` features, data is bucketed into 5 bins by default.
   - For `categorical` features, data is bucketed by category.
4. Test result is returned under the list of `data_drift` tests.

## Monitor Output

```JSON
{
    "data_drift":[
        {
            "test_name": "Kullback-Leibler",
            "test_category": "data_drift",
            "test_type": "kullback_leibler",
            "metric": "divergence",
            "test_id": "data_drift_kullback_leibler_divergence",
            "values": {
                <feature_1>: <divergence>,
                ...:...,
                <feature_n>: <divergence>
            },
        }
    ]
}
```