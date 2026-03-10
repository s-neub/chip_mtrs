# CHIP Flat File Data Ingestion

This document describes how to supply and update flat-file inputs for the CHIP preprocess pipeline and how the script selects which files to use.

## Inputs used by the pipeline

| Input | Purpose | Default (if not using config) |
|-------|---------|-------------------------------|
| **Batch activity log** | Human activity events; used for ground truth and batch enrichment | `batch_activity_log_202603042226.json` |
| **AI feedback** | AI correction feedback; used for ground truth | `ai_feedback_202603042225.json` |
| **AI Responses** | Directory of Claude AI response JSONs; flattened and merged | `AI Responses` |

The pipeline script is **CHIP_mtr_preprocess.py**. It can be run with explicit paths or with **latest-file selection** and **configurable source paths** via `config.yaml`.

## Where to put files

- **Same directory as the script (repo root):** Place `batch_activity_log_*.json` and `ai_feedback_*.json` in the project root (or in a subdirectory you configure).
- **AI Responses:** Keep all AI response JSON files in a single directory (default: `AI Responses`). The script reads every `.json` in that directory and flattens them.

## Naming convention

- **Activity logs:** `batch_activity_log_<suffix>.json` (e.g. `batch_activity_log_202603042226.json`). The script expects a top-level key `batch_activity_log` (array of records).
- **Feedback:** `ai_feedback_<suffix>.json` (e.g. `ai_feedback_202603042225.json`). The script expects a top-level key `ai_feedback` (array of records).
- **AI Responses:** Any `.json` in the AI Responses directory; structure is inferred (see preprocess docstrings).

## How the script selects input

1. **Explicit paths:** If you call `execute_pipeline(activity_file='path/to/file.json', feedback_file='path/to/feedback.json', ...)`, those paths are used as-is (unless overridden by config below).
2. **Config-based latest-file selection:** In `config.yaml` you can set a `sources` section:
   - `activity_directory`, `activity_pattern` (e.g. `"batch_activity_log_*.json"`) → the script uses **the latest file by modification time** in that directory matching the pattern.
   - `feedback_directory`, `feedback_pattern` (e.g. `"ai_feedback_*.json"`) → same for feedback.
   - `ai_responses_dir` → overrides the AI Responses directory.
3. **Latest-file helper:** The script uses `get_latest_flat_file(directory, pattern)`, which picks the file with the most recent mtime. No need to change code when you drop a new export; ensure the new file matches the pattern and has a newer timestamp than the previous one.

## Baseline–comparator split

The pipeline splits merged records into **baseline** and **comparator** using `ai_verification_time` (DATE split) or by record count (VOLUME split). For DATE split, the cutoff is chosen as follows:

- **Default (no `days_threshold` or config):** **Data-driven** — the script finds the first date such that baseline has at least `min_records_baseline` (default 20) and comparator at least `min_records_comparator` (default 20). The `split_method` column is set to `date-auto`.
- **Explicit `days_threshold`** (e.g. 30): Cutoff = max date − N days (`date-30`). If that would give an **empty baseline**, the script falls back to the data-driven cutoff and uses `date-auto`, and prints a one-line message.
- **Config `split.baseline_fraction`** (e.g. 0.4): **Percentile-based** — cutoff = min date + (max − min) × fraction, so the first 40% of the time range is baseline. `split_method` is `percentile-0.4`.

You can override split behavior via **`config.yaml`** under a `split` section:

```yaml
split:
  days_threshold: null    # null = data-driven; set to e.g. 30 for "last 30 days = comparator"
  min_records_baseline: 20
  min_records_comparator: 20
  baseline_fraction: null # optional; 0.0–1.0 for percentile-based split (e.g. 0.4)
```

If `split` is omitted, DATE split uses data-driven cutoff by default so both baseline and comparator get enough records when the data allow it.

## Step-by-step: adding or updating flat files

1. **Export or copy** the new activity log and/or feedback JSON into the project (or the directory specified in `sources.activity_directory` / `sources.feedback_directory`).
2. **Naming:** Use the naming convention above so the pattern (e.g. `batch_activity_log_*.json`) matches.
3. **Config (optional):** In `config.yaml`, add or update:
   ```yaml
   sources:
     activity_directory: "."          # or your data folder
     activity_pattern: "batch_activity_log_*.json"
     feedback_directory: "."
     feedback_pattern: "ai_feedback_*.json"
     ai_responses_dir: "AI Responses"
   ```
4. **Run the pipeline:** From the project root, run:
   ```bash
   python CHIP_mtr_preprocess.py
   ```
   Or call `execute_pipeline(...)` with the desired `split_method`, `days_threshold`, etc. The script will use the latest matching files when `sources` patterns are set.
5. **Verify outputs:** Check `CHIP_data/CHIP_master.csv` and `CHIP_mtr_1/`, `CHIP_mtr_2/`, `CHIP_mtr_3/` for updated baseline and comparator assets.

## Multi-file merge (optional)

For activity or feedback, if you have multiple files and want to merge them (last-wins per batch), you can use the helper `load_and_merge_with_upsert(file_paths, top_level_key, id_column='batchId')` in code. The current pipeline uses a single activity file and a single feedback file; to switch to multi-file upsert, you would collect matching paths (e.g. via `glob`), pass them to this helper, and use the returned DataFrame in place of loading one file in `derive_ground_truth` or `enrich_batch_dimensions`. This is optional and not wired by default.

## Monitor output and weight

**Output structure:** Each monitor’s `metrics()` returns a payload whose **keys are all emitted** so the ModelOp UI can display every element. New users can see what monitor outputs are available (e.g. `stability`, `data_drift`, `CSI_*`, `*_PSI`, `*_js_distance`, `generic_bar_graph`, `generic_table`, `generic_donut_chart`, and performance keys in M2) and configure the code behind each as needed. See each monitor’s `metrics()` docstring for the full list of output keys for that monitor.

**Weight column:** The pipeline adds a numeric column `weight` (default `1.0`) and the schema assigns only this column the weight role (`role: "weight"`, `dataClass: "numerical"`, `type: "float"`). No string or categorical column is used as weight. This keeps stability monitors (M1, M3) working and allows business logic to be added later.

**Designing the weight from business logic:** You can change the value of `weight` per record based on conditions. Example: increase weight when a record is more critical (e.g. remarks/comments indicate something not aligned with company policy, or a flag like `requires_escalation` or `high_risk_batch` is set). Implement by computing `weight` in the preprocess (or an upstream step) before the split: e.g. `weight = 2.0` when the condition is met, else `1.0`. The monitors then receive the dataframe with the desired per-record weights; no quantitative business rules are required during preproduction, but the structure is in place for when you define them.

## Path to S3 post-UAT

After UAT, data can be moved to S3. To point the pipeline at S3:

- **Option A:** Download the latest activity/feedback files from S3 to a local directory and run the pipeline as above, with `sources.activity_directory` / `sources.feedback_directory` set to that directory (or use a job that syncs from S3 to that path before running).
- **Option B:** If your runtime supports it, set `sources.activity_directory` (or paths in code) to an S3 path **only if** the implementation of `get_latest_flat_file` and `open()` are extended to support S3 (e.g. via `s3fs` or boto3). Out of the box, the script uses local filesystem paths.

Document the chosen S3 bucket and prefix (e.g. `s3://your-bucket/chip/inputs/`) in your runbooks so that post-UAT ingestion uses the same location.
