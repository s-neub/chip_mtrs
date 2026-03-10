# CHIP_MTR_3 Monitor

Tracks HITL reviewer calibration drift, intervention patterns, and decision stability over time.

## What this monitor tells you
- Tracks reviewer calibration drift and stability changes over time.
- Compares reviewer behavior to team average and summarizes QA feedback volume.

## Required Assets
- **Baseline Data:** Historical records used to establish expected behavior.
- **Comparator Data:** Recent production records used for current evaluation.
- **Schema Asset:** Parsed by `infer.validate_schema()` to map score/label/predictor roles.

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

