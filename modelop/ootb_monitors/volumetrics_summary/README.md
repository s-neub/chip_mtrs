# Volumetric Monitor: Summary
This ModelOp Center monitor returns a summary (min, max, standard deviation, etc.) of an asset.

## Input Assets

| Type          | Number | Description                                           |
| ------------- | ------ | ----------------------------------------------------- |
| Baseline Data | **0**  | |
| Comparator Data   | **1**  | Dataset to analyze |

## Assumptions & Requirements
 - Underlying `BUSINESS_MODEL` being monitored has a **job json** asset.
 - Input data contains at least one `numerical` column or one `categorical` column.

## Execution
1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
2. `metrics` function instantiates the **Volumetrics Monitor** class and uses the job json asset to determine the `numerical_columns` and `categorical_columns` accordingly.
3. The **summary** volumentrics test is run.
4. Test results are returned under the list of `volumetrics` tests.

## Monitor Output

```JSON
{
    "volumetrics":[
        {
            "test_name": "Summary",
            "test_category": "volumetrics",
            "test_type": "summary",
            "test_id": "volumetrics_summary",
            "values": {
                "numerical_summary": {
                    <numerical_feature_1>: {
                        "count": <count>,
                        "mean": <mean>,
                        "std": <standard_deviation>,
                        "min": <min>,
                        "25%": <first_quantile>,
                        "50%": <second_quantile>,
                        "75%": <third_quantile>,
                        "max": <max>
                    },
                    ...:...,
                    <numerical_feature_n>: {
                        "count": <count>,
                        "mean": <mean>,
                        "std": <standard_deviation>,
                        "min": <min>,
                        "25%": <first_quantile>,
                        "50%": <second_quantile>,
                        "75%": <third_quantile>,
                        "max": <max>
                    }
                },
                "categorical_summary": {
                    <categorical_feature_1>: {
                        "count": <count>,
                        "unique": <number_of_unique_values>,
                        "top": <mode>,
                        "freq": <frequency>
                    },
                    ...:...,
                    <categorical_feature_n>: {
                        "count": <count>,
                        "unique": <number_of_unique_values>,
                        "top": <mode>,
                        "freq": <frequency>
                    }
                }
            }
        }
    ]
}
```