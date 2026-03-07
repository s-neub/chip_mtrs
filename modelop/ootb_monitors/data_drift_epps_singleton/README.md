# Data Drift Monitor: Epps-Singleton
This ModelOp Center monitor computes **Epps-Singleton p-values** on **input** data.

## Input Assets

| Type          | Number | Description                                           |
| ------------- | ------ | ----------------------------------------------------- |
| Baseline Data | **1**  | A dataset corresponding to training/reference data    |
| Comparator Data   | **1**  | A dataset corresponding to a slice of production data |

## Assumptions & Requirements
 - Underlying `BUSINESS_MODEL` being monitored has a **job json** asset.
 - Input data contains at least one `numerical` column or one `categorical` column.

## Execution
1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
2. `metrics` function instantiates the **Data Drift Monitor** class and uses the job json asset to determine the `numerical_columns` and/or `categorical_columns` accordingly.
3. The **Epps-Singleton** data drift test is run.
4. Test result is returned under the list of `data_drift` tests.

## Monitor Output

```JSON
{
    "data_drift":[
        {
            "test_name": "Epps-Singleton",
            "test_category": "data_drift",
            "test_type": "epps_singleton",
            "metric": "p_value",
            "test_id": "data_drift_epps_singleton_p_value",
            "values": {
                <feature_1>: <p-value>,
                ...:...,
                <feature_n>: <p-value>
            },
        }
    ]
}
```