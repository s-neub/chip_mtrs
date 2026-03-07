# Bias Monitor: Disparity Metrics
This ModelOp Center monitor computes **disparity** metrics (with respect to reference groups) on **protected classes**, such as **race** or **gender**.

## Input Assets

| Type | Number | Description |
| ------ | ------ | ------ |
| Baseline Data | **0** | |
| Comparator Data | **1** |  A dataset corresponding to a slice of production data |

## Assumptions & Requirements
 - Underlying `BUSINESS_MODEL` being monitored has a **job json** asset.
 - `BUSINESS_MODEL` is a **classification** model.
 - Protected classes under consideration are **categorical** features.
 - Input data must contain:
     - 1 column with **role=label** (ground truth) 
     - 1 column with **role=score** (model output) 
     - At least 1 column with **protected_class=true** (protected attribute).

## Execution
1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
3. `metrics` function instantiates the **Bias Monitor** class and uses the job json asset to set the `protected_classes`, `label_column`, and `score_column`.
3. The **Aequitas Bias** test is run for each protected class in the list of protected classes. A reference group for each protected class can be provided in the job parameters. If no reference group is provided, the default behavior is to use all unique values as reference groups per protected class.
4. Test results are appended to the list of `bias` tests to be returned by the model.

## Monitor Output

```JSON
{
    "bias":[
        {
            "test_name": "Aequitas Bias",
            "test_category": "bias",
            "test_type": "bias",
            "protected_class": <protected_class_1>,
            "test_id": "bias_bias_"<protected_class_1>,
            "reference_group": <reference_group>,
            "thresholds": None,
            "values": [
                {
                    "attribute_name": <protected_class_1>,
                    "attribute_value": <reference_group>,
                    "ppr_disparity": 1.0,
                    "pprev_disparity": 1.0,
                    "precision_disparity": 1.0,
                    "fdr_disparity": 1.0,
                    "for_disparity": 1.0,
                    "fpr_disparity": 1.0,
                    "fnr_disparity": 1.0,
                    "tpr_disparity": 1.0,
                    "tnr_disparity": 1.0,
                    "npv_disparity": 1.0
                },
                {
                    "attribute_name": <protected_class_1>,
                    "attribute_value": <group_1>,
                    "ppr_disparity": <ppr_disparity>,
                    "pprev_disparity": <pprev_disparity>,
                    "precision_disparity": <precision_disparity>,
                    "fdr_disparity": <fdr_disparity>,
                    "for_disparity": <for_disparity>,
                    "fpr_disparity": <fpr_disparity>,
                    "fnr_disparity": <fnr_disparity>,
                    "tpr_disparity": <tpr_disparity>,
                    "tnr_disparity": <tnr_disparity>,
                    "npv_disparity": <npv_disparity>
                },
                ...,
                {
                    "attribute_name": <protected_class_1>,
                    "attribute_value": <group_n>,
                    "ppr_disparity": <ppr_disparity>,
                    "pprev_disparity": <pprev_disparity>,
                    "precision_disparity": <precision_disparity>,
                    "fdr_disparity": <fdr_disparity>,
                    "for_disparity": <for_disparity>,
                    "fpr_disparity": <fpr_disparity>,
                    "fnr_disparity": <fnr_disparity>,
                    "tpr_disparity": <tpr_disparity>,
                    "tnr_disparity": <tnr_disparity>,
                    "npv_disparity": <npv_disparity>
                }
            ],
        },
        {
            <aequitas_bias_test_result> for <protected_class_2>
        },
        ...,
        {
            <aequitas_bias_test_result> for <protected_class_n>
        }
    ]
}
```