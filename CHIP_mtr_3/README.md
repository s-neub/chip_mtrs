# CHIP_MTR_3 Monitor

Tracks Human QA behavior drift and human intervention volume changes.

## Required Assets
- **Baseline Data:** Historical dataset for establishing the baseline.
- **Comparator Data:** Production dataset to be evaluated.
- **Schema Asset:** Used by `infer.validate_schema()` to identify role assignments.

## Full-dimensional data
Baseline and comparator contain all batch-related columns. This monitor may pre-filter to the columns it needs in `init`/`metrics`.

## Output metrics and visualizations

- **Stability / drift:** CSI, PSI, and Jensen–Shannon distance by feature; bar charts, scatter (CSI vs JS), and summary table (Max/Min CSI, Score PSI).
- **HITL QA decision:** Donut and pie charts for comparator decision counts.
- **Reviewer vs team:** Per-reviewer volume and rejection rate vs team average:
  - **`reviewer_stats_table`:** List of `{Reviewer, Volume, Rejection Rate, vs Team}` (one row per reviewer). Positive “vs Team” means the reviewer rejects more often than the team average.
  - **`generic_table`:** Includes “Team Avg Rejection Rate”, “Reviewer &lt;ID&gt; Rejection Rate”, and “Reviewer &lt;ID&gt; vs Team” under feature “Reviewer Analysis”.
- **Time series:** **`time_line_graph`** shows daily rejection rate and review volume over time (dates from `hitl_review_time` or `ai_verification_time`). Series: “Rejection Rate” and “Volume” by date.
- String/categorical features (e.g. `activity_categories_seen`, `feedback_actions`) are included in stability and drift analysis and appear in the bar and scatter charts when present.
