# CHIP_MTR_3 Monitor

Tracks HITL reviewer calibration drift, intervention patterns, and decision stability over time.

## What this monitor tells you
- Tracks reviewer calibration drift and stability changes over time.
- Compares reviewer behavior to team average and summarizes QA feedback volume.

## Required Assets
- **Baseline Data:** Historical records used to establish expected behavior.
- **Comparator Data:** Recent production records used for current evaluation.
- **Test Results Output:** `CHIP_mtr_3_test_results.json` written by local smoke runs.

## ModelOp UI File Roles
- Set `CHIP_mtr3_hitl_stability.py` as both **Model Source** and **Primary Source**.
- Set `README.md` as **Readme File**.
- Set `required_assets.json` as the required-assets specification file.
- Do not assign `*_test_results.json`, `*_error.txt`, `*.dmn`, `modelop_schema.json`, or `blank_schema_asset.csv` as runtime input data assets.

## Runtime Initialization Contract
- Entry points are `init(job_json)` and `metrics(df_baseline, df_sample)`.
- `init()` stores the runtime `JOB` object, validates schema via `infer.validate_schema(job_json)`, parses `rawJson.jobParameters`, and captures `referenceModel.group` into `GROUP`.
- Optional job parameters:
  - `AI_FAIL_VALUES` (default: `["FAIL"]`) for AI status mapping.
  - `HITL_POSITIVE_VALUES` (default: `["REJECTED", "REPROCESS", "PENDING"]`) for HITL decision mapping.
  - `M3_TOP_N_FEATURES` (default: `20`) to cap plotted feature count.
- Runtime assets in this folder:
  - baseline: `CHIP_mtr_3_baseline.json` / `CHIP_mtr_3_baseline.csv`
  - comparator: `CHIP_mtr_3_comparator.json` / `CHIP_mtr_3_comparator.csv`
  - test results: `CHIP_mtr_3_test_results.json`

## UI Output Interpretation
- **Generic Table:** Largest/smallest CSI, overall PSI, team vs reviewer deltas, QA sample count, date windows.
- **Generic Bar Graph / Horizontal Bar Graph:** CSI (`data1`) and JS distance (`data2`) by feature.
- **Generic Scatter Plot:** Feature-level CSI vs JS relationship.
- **Time Line Graph:** Daily rejection rate (`data1`) and review volume (`data2`).
- **Generic Pie/Donut:** Comparator HITL decision mix.

## Data Notes
- Baseline and comparator exports are full-dimensional batch data.
- Monitor scripts pre-filter columns needed by their metrics functions.
- `CHIP_data/CHIP_master.*` is always refreshed by preprocess runs.

