# CHIP_MTR_1 Monitor

Tracks the behavior and drift of the Claude AI model's output over time using PSI and Data Drift methods.

## Required Assets
- **Baseline Data:** Historical dataset for establishing the baseline.
- **Comparator Data:** Production dataset to be evaluated.
- **Schema Asset:** Used by `infer.validate_schema()` to identify role assignments.

## Full-dimensional data
Baseline and comparator contain all batch-related columns. This monitor may pre-filter to the columns it needs in `init`/`metrics`.
