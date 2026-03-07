# Volumetric Monitor: Count
This ModelOp Center monitor returns the record count of an asset.

## Input Assets

| Type          | Number | Description                                           |
| ------------- | ------ | ----------------------------------------------------- |
| Baseline Data | **0**  | |
| Comparator Data   | **1**  | Dataset to count |

## Assumptions & Requirements

## Execution
1. `metrics` function runs a **count** volumetrics test.
2. Test result is returned under the list of `volumetrics` tests.

## Monitor Output

```JSON
{
    "record_count": <dataframe_record_count>,

    "volumetrics":[
        {
            "test_name": "Count",
            "test_category": "volumetrics",
            "test_type": "count",
            "test_id": "volumetrics_count",
            "values": {
                "record_count": <dataframe_record_count>
            }
        }
    ]
}
```