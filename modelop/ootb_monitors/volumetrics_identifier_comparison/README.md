# Volumetric Monitor: Identifier Comparison
This ModelOp Center monitor detects discrepancies between two assets based on their record identifiers.

## Input Assets

| Type          | Number | Description                                           |
| ------------- | ------ | ----------------------------------------------------- |
| Baseline Data | **1**  | First dataset to compare |
| Comparator Data   | **1**  | Second dataset to compare |

## Assumptions & Requirements
 - Underlying `BUSINESS_MODEL` being monitored has a **job json** asset.
 - Input data must contain:
     - At least 1 column with **role=identifier**

## Execution
1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
2. `metrics` function instantiates the **Volumetrics Monitor** class and uses the job json asset to determine the `identifier_columns` accordingly.
3. The **identifier_comparison** test is run.
4. Test results are returned under the list of `volumetrics` tests.

## Monitor Output

```JSON
{
    "identifiers_match": <boolean>,

    "volumetrics":[
        {
            "test_name": "Identifier Comparison",
            "test_category": "volumetrics",
            "test_type": "identifier_comparison",
            "test_id": "volumetrics_identifier_comparison",
            "values": {
                "identifiers_match": <boolean>,
                "dataframe_1": {
                    "identifier_columns": <id_columns_array>,
                    "record_count": <dataframe_1_record_count>,
                    "unique_identifier_count": <dataframe_1_unique_identifier_count>,
                    "extra_identifiers": {
                        "total": <number_of_extra_identifiers>,
                        "breakdown": <breakdown_of_extra_identifiers>
                    }
                },
                "dataframe_2": {
                    "identifier_columns": <id_columns_array>,
                    "record_count": <dataframe_1_record_count>,
                    "unique_identifier_count": <dataframe_2_unique_identifier_count>,
                    "extra_identifiers": {
                        "total": <number_of_extra_identifiers>,
                        "breakdown": <breakdown_of_extra_identifiers>
                    }
                }
            }
        }
    ]
}
```