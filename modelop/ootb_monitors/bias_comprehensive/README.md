# Bias Monitor: Disparity and Group Metrics

This ModelOp Center monitor computes **disparity** metrics (with respect to reference groups) and **group** metrics on **protected classes**, such as **race** or **gender**.

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
3. The **Aequitas Bias** test and an **Aequitas Group** test for each protected class in the list of protected classes. A reference group for each protected class is chosen by default (first occurence).
4. Test results are appended to the list of `bias` tests to be returned by the model.

## Monitor Output

```JSON
{
    "bias":[
        <aequitas_bias_test_result>, <aequitas_group_test_result> for protected_class in protected_classes
    ]
}
```