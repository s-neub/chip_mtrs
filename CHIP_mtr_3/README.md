# CHIP_MTR_3 Monitor

Tracks Human QA behavior drift (e.g., rubber-stamping detection) and human intervention volume changes.

## Required Assets
- **Baseline Data:** Historical dataset for establishing the baseline.
- **Comparator Data:** Production dataset to be evaluated.
- **Schema Asset:** Used by `infer.validate_schema()` to identify role assignments.

## Execution
1. The `init` function reads the schema asset to identify predictors, scores, and labels.
2. The `metrics` function computes the test results and yields the JSON payload.
