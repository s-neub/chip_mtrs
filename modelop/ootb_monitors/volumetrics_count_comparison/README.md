# Volumetric Monitor: Count Comparison
This ModelOp Center monitor detects discrepancies between two assets based on their record counts.

## Input Assets

| Type          | Number | Description                                           |
| ------------- | ------ | ----------------------------------------------------- |
| Baseline Data | **1**  | First dataset to compare |
| Comparator Data   | **1**  | Second dataset to compare |

## Assumptions & Requirements

## Execution
1. `metrics` function runs a **count_comparison** volumetrics test.
2. Test result is returned under the list of `volumetrics` tests.

## Monitor Output

```JSON
{
    "record_count_difference": <record_count_difference>,

    "volumetrics":[
        {
            "test_name": "Count Comparison",
            "test_category": "volumetrics",
            "test_type": "count_comparison",
            "test_id": "volumetrics_count_comparison",
            "values": {
                "dataframe_1_record_count": <dataframe_1_record_count>,
                "dataframe_2_record_count": <dataframe_2_record_count>,
                "record_count_difference": <record_count_difference>
            }
        }
    ]
}
```