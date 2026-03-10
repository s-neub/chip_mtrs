# CHIP_MTR_1 Monitor

Tracks AI output stability and drift between baseline and comparator windows.

## What this monitor tells you
- Detects shifts between baseline and comparator for AI outputs and related dimensions.
- Highlights which features are most unstable (CSI) and how that aligns with drift distance (JS).

## Required Assets
- **Baseline Data:** Historical records used to establish expected behavior.
- **Comparator Data:** Recent production records used for current evaluation.
- **Schema Asset:** Parsed by `infer.validate_schema()` to map score/label/predictor roles.
- **Test Results Output:** `CHIP_mtr_1_test_results.json` written by local smoke runs.

## Runtime Initialization Contract
- Entry points are `init(job_json)` and `metrics(df_baseline, df_sample)`.
- `init()` stores the runtime `JOB` object, validates schema via `infer.validate_schema(job_json)`, and parses `rawJson.jobParameters` with defaults.
- Optional job parameters:
  - `AI_FAIL_VALUES` (default: `["FAIL"]`) for converting AI status to numeric score.
  - `M1_TOP_N_FEATURES` (default: `20`) to cap plotted feature count.
- Runtime assets in this folder:
  - baseline: `CHIP_mtr_1_baseline.json` / `CHIP_mtr_1_baseline.csv`
  - comparator: `CHIP_mtr_1_comparator.json` / `CHIP_mtr_1_comparator.csv`
  - test results: `CHIP_mtr_1_test_results.json`

## UI Output Interpretation
- **Generic Table:** High-level summary (largest/smallest CSI, overall PSI, date windows).
- **Generic Bar Graph / Horizontal Bar Graph:** Side-by-side CSI (`data1`) and JS distance (`data2`) by feature.
- **Generic Scatter Plot:** CSI vs JS relationship across top features.
- **Generic Pie/Donut:** Comparator AI outcome mix.

## Data Notes
- Baseline and comparator exports are full-dimensional batch data.
- Monitor scripts pre-filter columns needed by their metrics functions.
- `CHIP_data/CHIP_master.*` is always refreshed by preprocess runs.

