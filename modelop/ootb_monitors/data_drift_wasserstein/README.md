# Data Drift Monitor: Wasserstein
This ModelOp Center monitor computes the **Wasserstein Distance** on **input** data.

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
2. `metrics` function instantiates the **Data Drift Monitor** class and uses the job json asset to determine the `numerical_columns`.
3. The **Wasserstein** concept drift test is run.
4. Test result is returned under the list of `data_drift` tests.

## Monitor Output

```JSON
{
    "data_drift":[
        {
            "test_name": "Wasserstein",
            "test_category": "data_drift",
            "test_type": "wasserstein",
            "metric": "distance",
            "test_id": "data_drift_wasserstein_distance",
            "values": {
                <feature_1>: <distance>,
                ...:...,
                <feature_n>: <distance>
            },
        }
    ]
}
```