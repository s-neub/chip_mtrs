import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
# The 'modelop.utils' import is commented out as it's not used in this script.
# If you are in a ModelOp environment, you might uncomment it and use 'utils.configure_logger()'.
# import modelop.utils as utils


def _safe_divide(numerator, denominator, nan_val=np.nan):
    """
    Prevents division by zero or division involving NaN values.

    In metric calculations (like Precision or Recall), the denominator can
    sometimes be zero (e.g., no predicted positives). Standard division
    would cause a ZeroDivisionError. This function catches such cases.

    Args:
        numerator (float or int): The number on top.
        denominator (float or int): The number on the bottom.
        nan_val (any, optional): The value to return if division is unsafe.
                                 Defaults to np.nan (Not a Number).

    Returns:
        float: The result of the division, or `nan_val` if division is unsafe.
    """
    # Check if denominator is 0 OR if either number is NaN
    if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
        return nan_val
    # If safe, perform the division
    return numerator / denominator

def _calculate_metrics(y_true, y_pred, requested_metrics):
    """
    Calculates all specified metrics from a confusion matrix for a given bin.
    
    This function is the core performance calculation engine. It takes the
    true labels and predicted labels for a specific time window (a "bin")
    and calculates metrics like Sensitivity, Specificity, etc.

    Args:
        y_true (pd.Series): A pandas Series of true ground truth labels (0s and 1s).
        y_pred (pd.Series): A pandas Series of predicted labels (0s and 1s).
        requested_metrics (list[str]): A list of metric abbreviations (e.g., ['SEN', 'SP']).

    Returns:
        dict: A dictionary with metric_abbr as key and calculated value as value.
              Example: {'SEN': 0.85, 'SP': 0.92}
    """
    # Initialize a dictionary to store the results
    results = {}

    # If there are no true labels for this bin, return NaN for all requested metrics
    if len(y_true) == 0:
        for metric in requested_metrics:
            results[metric] = np.nan
        return results

    # --- Confusion Matrix Calculation ---
    # A confusion matrix is the basis for most classification metrics.
    # It's a 2x2 table:
    #
    #                 Predicted: 0   Predicted: 1
    #     Actual: 0       TN             FP
    #     Actual: 1       FN             TP
    #
    try:
        # 'ravel()' flattens the 2x2 matrix into a 1D array [tn, fp, fn, tp]
        # We specify labels=[0, 1] to ensure the order is correct.
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        # This can happen if y_true or y_pred contains only one class (e.g., all 0s)
        # in a way that confusion_matrix can't handle.
        # In this case, we can't calculate metrics, so return NaNs.
        for metric in requested_metrics:
            results[metric] = np.nan
        return results

    # --- Calculate Basic Components ---
    # These are the sums derived from the confusion matrix elements
    available_metrics = {}
    P = tp + fn      # Total Actual Positives (or "Positives")
    N = tn + fp      # Total Actual Negatives (or "Negatives")
    PP = tp + fp     # Total Predicted Positives (or "Predicted Positives")
    PN = tn + fn     # Total Predicted Negatives (or "Predicted Negatives")
    Total = P + N    # Total number of observations

    # --- Calculate Individual Metrics ---
    # We use 'set().intersection()' as an optimization.
    # For example, Sensitivity (SEN) is needed to calculate F1, J, BM, etc.
    # This check ensures we only calculate SEN if 'SEN' OR 'F1' OR 'J' ... is requested.
    
    # Sensitivity (SEN) or Recall or True Positive Rate (TPR)
    # "Of all the *actual* positive cases, what fraction did we *correctly* predict as positive?"
    available_metrics['SEN'] = _safe_divide(tp, P) if set(['SEN','F1','INF','J','BM','PT','TS','CSI','DOR']).intersection(set(requested_metrics)) else None
    
    # Specificity (SP) or True Negative Rate (TNR)
    # "Of all the *actual* negative cases, what fraction did we *correctly* predict as negative?"
    available_metrics['SP'] = _safe_divide(tn, N) if set(['SP','INF','J','BM','PT','DOR']).intersection(set(requested_metrics)) else None
    
    # Precision (PPV) or Positive Predictive Value
    # "Of all the cases we *predicted* as positive, what fraction were *actually* positive?"
    available_metrics['PPV'] = _safe_divide(tp, PP) if set(['PPV','F1','MK']).intersection(set(requested_metrics)) else None
    
    # Negative Predictive Value (NPV)
    # "Of all the cases we *predicted* as negative, what fraction were *actually* negative?"
    available_metrics['NPV'] = _safe_divide(tn, PN) if set(['NPV','MK']).intersection(set(requested_metrics)) else None
    
    # False Negative Rate (FNR) or Miss Rate
    # "Of all the *actual* positive cases, what fraction did we *incorrectly* predict as negative?"
    available_metrics['FNR'] = _safe_divide(fn, P) if 'FNR' in requested_metrics else None
    
    # False Positive Rate (FPR)
    # "Of all the *actual* negative cases, what fraction did we *incorrectly* predict as positive?"
    available_metrics['FPR'] = _safe_divide(fp, N) if 'FPR' in requested_metrics else None
    
    # False Discovery Rate (FDR)
    # "Of all the cases we *predicted* as positive, what fraction were *actually* negative?"
    available_metrics['FDR'] = _safe_divide(fp, PP) if 'FDR' in requested_metrics else None
    
    # False Omission Rate (FOR)
    # "Of all the cases we *predicted* as negative, what fraction were *actually* positive?"
    available_metrics['FOR'] = _safe_divide(fn, PN) if 'FOR' in requested_metrics else None
    
    # Accuracy (ACC)
    # "Of all cases, what fraction did we predict correctly (both positive and negative)?"
    available_metrics['ACC'] = _safe_divide(tp + tn, Total) if 'ACC' in requested_metrics else None
    
    # Error Rate (ERR)
    # "Of all cases, what fraction did we predict incorrectly?" (Opposite of Accuracy)
    available_metrics['ERR'] = _safe_divide(fp + fn, Total) if 'ERR' in requested_metrics else None
    
    # F1-Score (F1)
    # The harmonic mean of Precision and Sensitivity. Good for imbalanced datasets.
    if 'F1' in requested_metrics and available_metrics['PPV'] is not None and available_metrics['SEN'] is not None:
        precision = available_metrics['PPV'] 
        sensitivity = available_metrics['SEN'] 
        available_metrics['F1'] = _safe_divide(2 * precision * sensitivity, precision + sensitivity)
    else:
        available_metrics['F1'] = None
    
    # Prevalence (PR)
    # "What fraction of the *actual* data was positive?"
    available_metrics['PR'] = _safe_divide(P, Total) if 'PR' in requested_metrics else None
    
    # Matthews Correlation Coefficient (MCC)
    # A robust metric that's only high if the model is good on all 4 confusion matrix cells.
    if 'MCC' in requested_metrics:
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        available_metrics['MCC'] = _safe_divide(mcc_num, mcc_den)
    else:
        available_metrics['MCC'] = None

    # Informedness (INF) or Youden's J (J) or Bookmaker's (BM)
    # Measures how "informed" a prediction is, beyond random chance. (SEN + SP - 1)
    available_metrics['INF'] = available_metrics['SEN'] + available_metrics['SP'] - 1 if set(['INF','J','BM']).intersection(set(requested_metrics)) and available_metrics['SEN'] is not None and available_metrics['SP'] is not None else None
    available_metrics['J'] = available_metrics['INF'] if 'J' in requested_metrics else None
    available_metrics['BM'] = available_metrics['INF'] if 'BM' in requested_metrics else None
    
    # Markedness (MK)
    # Measures the "markedness" of the prediction. (PPV + NPV - 1)
    available_metrics['MK'] = available_metrics['PPV'] + available_metrics['NPV'] - 1 if 'MK' in requested_metrics and available_metrics['PPV'] is not None and available_metrics['NPV'] is not None else None
    
    # Prevalence Threshold (PT)
    if 'PT' in requested_metrics and available_metrics['SEN'] is not None and available_metrics['SP'] is not None:
            pt_num = (np.sqrt(available_metrics['SEN'] * (1 - available_metrics['SP'])) + available_metrics['SP'] - 1)
            pt_den = (available_metrics['SEN'] + available_metrics['SP'] - 1)
            available_metrics['PT'] = _safe_divide(pt_num, pt_den)
    else:
        available_metrics['PT'] = None

    # Threat Score (TS) or Critical Success Index (CSI)
    # Measures how well "yes" events were predicted, ignoring correct "no"s.
    available_metrics['TS'] = _safe_divide(tp, tp + fn + fp) if set(['TS','CSI']).intersection(set(requested_metrics)) else None
    available_metrics['CSI'] = available_metrics['TS'] if 'CSI' in requested_metrics else None

    # Diagnostic Odds Ratio (DOR)
    # The ratio of the odds of a positive test in a diseased person vs. a non-diseased.
    if 'DOR' in requested_metrics:
        if tp == 0 or tn == 0 or fp == 0 or fn == 0:
            # DOR is undefined if any cell is 0, often set to NaN or handled with smoothing.
            available_metrics['DOR'] = np.nan
        else:
            dor_num = (tp * tn)
            dor_den = (fp * fn)
            available_metrics['DOR'] = _safe_divide(dor_num, dor_den)

    # --- Final Output ---
    # Populate the 'results' dictionary with only the metrics that were requested.
    for metric in requested_metrics:
        results[metric] = available_metrics.get(metric, np.nan)

    return results

def _apply_metrics_to_bin(df_binned, requested_metrics):
    """
    Pandas .apply() helper function.
    
    This function is designed to be used with `pandas.DataFrame.resample(...).apply()`.
    Pandas will "resample" the data into time bins (e.g., all data for 'Week 1',
    all data for 'Week 2', ...).
    For each bin, it passes the sub-dataframe (e.g., all rows for 'Week 1')
    to this function. This function then computes the metrics for that small
    dataframe.

    Args:
        df_binned (pd.DataFrame): The sub-dataframe for one time bin.
        requested_metrics (list[str]): List of metric abbreviations to calculate.

    Returns:
        pd.Series: A Series of calculated metrics (e.g., index=['SEN', 'SP'], data=[0.85, 0.92])
    """
    # If the bin is empty, return a Series of NaNs
    if df_binned.empty:
        return pd.Series(index=requested_metrics, data=np.nan, dtype=float)

    # Call the main calculation function on the binned data
    metrics_dict = _calculate_metrics(df_binned['y_true'],
                                      df_binned['y_pred'],
                                      requested_metrics)
    
    # Convert the results dictionary to a pandas Series, which pandas .apply() expects
    return pd.Series(metrics_dict)

def calculate_binned_metrics(df, timestamp_col, bins, 
                             label_col, label_true, label_false,
                             score_col, score_true, score_false,
                             metrics=None, numeric_aggregations=None, logger=None):
    """
    Calculates specified performance metrics and/or numeric aggregations
    over different time bins.

    This is the main "universal" function that can be used outside of ModelOp.
    It takes a DataFrame, configuration, and returns a dictionary of
    binned calculations.

    Args:
        df (pd.DataFrame): The input DataFrame containing predictions and actuals.
        timestamp_col (str): The name of the column in 'df' that contains
                             the datetime information (e.g., 'Date').
        bins (list[str]): A list of pandas resampling frequency strings.
                          Examples: ['W'] (Weekly), ['MS'] (Month-Start), ['YS'] (Year-Start)
        label_col (str): The name of the "actual" (ground truth) column.
        label_true (any): The value in 'label_col' representing a true (1) case.
        label_false (any): The value in 'label_col' representing a false (0) case.
        score_col (str): The name of the "predicted" (score) column.
        score_true (any): The value in 'score_col' representing a true (1) prediction.
        score_false (any): The value in 'score_col' representing a false (0) prediction.
        metrics (list[str], optional): A list of metric abbreviations to calculate.
        numeric_aggregations (dict, optional): A dictionary specifying numeric columns
                                     and the aggregation to perform.
                                     Format: {'col_name': 'agg_function'}
                                     Example: {'loan_amount': 'mean', 'age': 'median'}
        logger (logging.Logger, optional): Logger for warnings.

    Returns:
        dict: A dictionary where keys are the binning frequencies (e.g., 'W')
              and values are DataFrames containing the calculated metrics
              and/or aggregations.
    """
    
    # --- Setup ---
    # Create a copy to avoid modifying the original DataFrame (SettingWithCopyWarning)
    df_copy = df.copy()
    
    # Use the provided logger, or default to the standard 'print' function if no logger
    log_func = logger.warning if logger else print
    
    # If no work is requested, return an empty dictionary
    if not metrics and not numeric_aggregations:
        log_func("Warning: No 'metrics' or 'numeric_aggregations' specified. Returning empty results.")
        return {bin_freq: pd.DataFrame() for bin_freq in bins}

    # --- Performance Metrics Setup (if requested) ---
    if metrics:
        log_func("Setting up for performance metric calculation...")
        
        # Check if the necessary column names were provided
        if not label_col or not score_col:
            log_func("Error: 'label_col' and 'score_col' are required when 'metrics' are requested.")
            raise ValueError("'label_col' and 'score_col' are required when 'metrics' are requested.")

        # Check if the columns actually exist in the DataFrame
        # This is where the original KeyError likely came from
        if label_col not in df_copy.columns:
            log_func(f"Error: Label column '{label_col}' not found in DataFrame.")
            raise KeyError(f"Label column '{label_col}' not found in DataFrame columns: {list(df_copy.columns)}")
        
        if score_col not in df_copy.columns:
            log_func(f"Error: Score column '{score_col}' not found in DataFrame.")
            raise KeyError(f"Score column '{score_col}' not found in DataFrame columns: {list(df_copy.columns)}")

        # --- Map values to 0s and 1s using np.where (alternative to .map()) ---
        # This method avoids issues with older pandas versions (like 1.2.3)
        # It creates a new column 'y_true' where:
        # 1. If df_copy[label_col] == label_true, set to 1
        # 2. Else if df_copy[label_col] == label_false, set to 0
        # 3. Otherwise, set to np.nan (Not a Number)
        
        log_func(f"Mapping actual column '{label_col}' using '{label_true}': 1, '{label_false}': 0")
        df_copy['y_true'] = np.where(
            df_copy[label_col] == label_true, 
            1, 
            np.where(df_copy[label_col] == label_false, 0, np.nan)
        )
        
        log_func(f"Mapping predicted column '{score_col}' using '{score_true}': 1, '{score_false}': 0")
        df_copy['y_pred'] = np.where(
            df_copy[score_col] == score_true,
            1,
            np.where(df_copy[score_col] == score_false, 0, np.nan)
        )

        # --- Check for Unmapped Values ---
        # If the data contains values not specified (e.g., a 'Maybe' string),
        # the np.where logic will produce 'NaN' (Not a Number).
        if df_copy['y_true'].isnull().any():
            # Find the original values that resulted in NaN
            unmapped_vals = df_copy[df_copy['y_true'].isnull()][label_col].unique()
            log_func(f"Warning: Found unmapped values in actual column '{label_col}': {unmapped_vals}. "
                     f"These rows will be ignored for metric calculations.")
        if df_copy['y_pred'].isnull().any():
            # Find the original values that resulted in NaN
            unmapped_vals = df_copy[df_copy['y_pred'].isnull()][score_col].unique()
            log_func(f"Warning: Found unmapped values in predicted column '{score_col}': {unmapped_vals}. "
                     f"These rows will be ignored for metric calculations.")
        
        # Create a new, temporary DataFrame that *only* contains rows where
        # mapping was successful (both y_true and y_pred are not NaN).
        metrics_df_copy = df_copy.dropna(subset=['y_true', 'y_pred'])
        
        # If *no* rows are left after mapping, we can't calculate metrics.
        if metrics_df_copy.empty and not numeric_aggregations:
            log_func("Warning: No valid data remaining after mapping for metrics. Returning empty results.")
            return {bin_freq: pd.DataFrame(columns=metrics) for bin_freq in bins}
            
        # Ensure the 0s and 1s are treated as integers
        metrics_df_copy['y_true'] = metrics_df_copy['y_true'].astype(int)
        metrics_df_copy['y_pred'] = metrics_df_copy['y_pred'].astype(int)
    else:
        # If no metrics are requested, create an empty DataFrame to skip calculations.
        metrics_df_copy = pd.DataFrame() 

    # --- Timestamp Setup ---
    try:
        log_func(f"Converting timestamp column '{timestamp_col}' to datetime objects.")
        # We set the index on the *original* df_copy for aggregations...
        df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])
        df_copy = df_copy.set_index(timestamp_col)
        
        # ...and *also* on the metrics_df_copy for metric calculations.
        if not metrics_df_copy.empty:
            # This conversion might be redundant if metrics_df_copy is just a slice,
            # but it's safer to be explicit.
            metrics_df_copy[timestamp_col] = pd.to_datetime(metrics_df_copy[timestamp_col])
            metrics_df_copy = metrics_df_copy.set_index(timestamp_col)
            
    except Exception as e:
        log_func(f"Error: Could not parse timestamp column '{timestamp_col}'. Error: {e}")
        raise ValueError(f"Could not parse timestamp column '{timestamp_col}'. "
                         f"Ensure it is a valid datetime format. Error: {e}")

    # --- Binned Calculation Loop ---
    log_func(f"Starting calculation loop for bins: {bins}")
    all_binned_metrics = {}
    
    # Loop through each requested bin frequency (e.g., 'W', 'MS', 'YS')
    for bin_freq in bins:
        # This list will hold the results for this bin (e.g., metrics_df, agg_df)
        bin_results_dfs = [] 
        
        # --- 1. Calculate Performance Metrics (if requested) ---
        if metrics:
            if metrics_df_copy.empty:
                 log_func(f"Warning: 'metrics' were requested but no valid data rows were found after mapping. Skipping performance metrics for bin '{bin_freq}'.")
            else:
                try:
                    log_func(f"Resampling performance metrics for bin '{bin_freq}'...")
                    # This is the core 'resample' operation.
                    # 1. .resample(bin_freq): Groups data by time (e.g., by week).
                    # 2. .apply(...): For each group, calls our '_apply_metrics_to_bin' function.
                    
                    # --- FIX: Use a lambda function to pass extra arguments to .apply() ---
                    # This resolves the type-hinting warning. The lambda 'x'
                    # represents the 'df_binned' DataFrame that .apply() passes in.
                    metrics_df = metrics_df_copy.resample(bin_freq).apply(
                        lambda x: _apply_metrics_to_bin(x, requested_metrics=metrics)
                    )
                    # Drop any bins that were all NaN (e.g., a week with no data)
                    metrics_df = metrics_df.dropna(how='all')
                    
                    if not metrics_df.empty:
                        bin_results_dfs.append(metrics_df)
                        
                except ValueError as e:
                    # This catches bad bin_freq strings (e.g., 'Weekly' instead of 'W')
                    log_func(f"Warning: Invalid bin frequency string '{bin_freq}' for metrics. Skipping. Error: {e}")
                except Exception as e:
                    log_func(f"An unexpected error occurred during metrics resampling with bin '{bin_freq}': {e}")

        # --- 2. Calculate Numeric Aggregations (if requested) ---
        if numeric_aggregations:
            try:
                # Check which requested aggregation columns actually exist in the DataFrame
                valid_agg_cols = {col: agg for col, agg in numeric_aggregations.items() if col in df_copy.columns}
                missing_cols = set(numeric_aggregations.keys()) - set(valid_agg_cols.keys())
                
                if missing_cols:
                    log_func(f"Warning: For bin '{bin_freq}', the following columns for numeric aggregation were not found and will be skipped: {missing_cols}")

                if valid_agg_cols:
                    log_func(f"Resampling numeric aggregations for bin '{bin_freq}': {valid_agg_cols}")
                    # Use the original df_copy (now indexed) for aggregations
                    # 1. .resample(bin_freq): Groups data by time.
                    # 2. .agg(...): For each group, applies standard functions like 'mean', 'median', 'sum'.
                    agg_df = df_copy.resample(bin_freq).agg(valid_agg_cols)
                    
                    # Drop any bins that were all NaN
                    agg_df = agg_df.dropna(how='all')
                    if not agg_df.empty:
                        bin_results_dfs.append(agg_df)
                else:
                    log_func(f"Warning: No valid columns found for numeric aggregation for bin '{bin_freq}'.")

            except (AttributeError, TypeError) as e:
                # This catches bad aggregation function names (e.g., 'avg' instead of 'mean')
                log_func(f"Warning: Invalid aggregation function provided in 'numeric_aggregations' for bin '{bin_freq}'. Skipping. Error: {e}")
            except Exception as e:
                log_func(f"An unexpected error occurred during numeric aggregation with bin '{bin_freq}': {e}")

        # --- 3. Combine and Store Results for the bin ---
        if not bin_results_dfs:
            # If no data was produced for this bin, store an empty DataFrame
            all_binned_metrics[bin_freq] = pd.DataFrame()
        else:
            # Combine the metrics_df and agg_df (if they exist) side-by-side.
            # axis=1 means "concatenate along columns". They join on the timestamp index.
            all_binned_metrics[bin_freq] = pd.concat(bin_results_dfs, axis=1)

    log_func("Binned calculations complete.")
    return all_binned_metrics


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- BEGIN MODELOP SCRIPT ---
# --- (This is the section ModelOp Center executes) ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# modelop.init
def init(init_param):
    """
    ModelOp init function. Sets global variables from job parameters.
    
    This function is called once when the ModelOp job starts.
    It reads configuration from the 'init_param' (which is a JSON string
    of job parameters) and sets global variables that the 'metrics'
    function will use.
    
    This version uses try/except blocks to load each parameter,
    allowing users to set them in the ModelOp UI. If a parameter is
    not set, it falls back to a hardcoded default value.
    """
    
    # Set up a basic logger. In a full ModelOp environment,
    # you might use 'logger = utils.configure_logger()'
    global logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Initializing binned metrics job...")

    # Parse the job parameters JSON string
    try:
        job = json.loads(init_param["rawJson"])
    except Exception as e:
        logger.warning(f"Could not parse 'rawJson' in init_param. Using all default values. Error: {e}")
        job = {} # Set to empty dict to allow defaults to work

    # --- Declare global variables ---
    # These variables will be accessible by the 'metrics' function
    global TIMESTAMP_COLUMN, LABEL_COLUMN_NAME, LABEL_FALSE_VALUE, \
           LABEL_TRUE_VALUE, SCORE_COLUMN_NAME, SCORE_FALSE_VALUE, \
           SCORE_TRUE_VALUE, METRICS_TO_CALC, BINS_TO_CALC, NUMERIC_AGGS_TO_CALC

    # --- Load Job Parameters with Try/Except ---
    
    # Parameter: TIMESTAMP_COLUMN
    try:
        TIMESTAMP_COLUMN = job["jobParameters"]["TIMESTAMP_COLUMN"]
        logger.info(f"Loaded TIMESTAMP_COLUMN from job parameters: '{TIMESTAMP_COLUMN}'")
    except Exception:
        TIMESTAMP_COLUMN = 'Date'
        logger.info(f"Using default TIMESTAMP_COLUMN: '{TIMESTAMP_COLUMN}'")

    # --- Actual/Label Parameters ---
    # NOTE: The 'KeyError: label' you experienced was likely because
    # the default 'label' column name did not exist in your input data.
    # By setting LABEL_COLUMN_NAME as a job parameter to match your
    # data, you can fix this.
    
    # Parameter: LABEL_COLUMN_NAME
    try:
        LABEL_COLUMN_NAME = job["jobParameters"]["LABEL_COLUMN_NAME"]
        logger.info(f"Loaded LABEL_COLUMN_NAME from job parameters: '{LABEL_COLUMN_NAME}'")
    except Exception:
        LABEL_COLUMN_NAME = 'label'
        logger.info(f"Using default LABEL_COLUMN_NAME: '{LABEL_COLUMN_NAME}'")

    # Parameter: LABEL_FALSE_VALUE
    try:
        LABEL_FALSE_VALUE = job["jobParameters"]["LABEL_FALSE_VALUE"]
        logger.info(f"Loaded LABEL_FALSE_VALUE from job parameters: '{LABEL_FALSE_VALUE}'")
    except Exception:
        LABEL_FALSE_VALUE = False
        logger.info(f"Using default LABEL_FALSE_VALUE: {LABEL_FALSE_VALUE}")
        
    # Parameter: LABEL_TRUE_VALUE
    try:
        LABEL_TRUE_VALUE = job["jobParameters"]["LABEL_TRUE_VALUE"]
        logger.info(f"Loaded LABEL_TRUE_VALUE from job parameters: '{LABEL_TRUE_VALUE}'")
    except Exception:
        LABEL_TRUE_VALUE = True
        logger.info(f"Using default LABEL_TRUE_VALUE: {LABEL_TRUE_VALUE}")

    # --- Predicted/Score Parameters ---
    
    # Parameter: SCORE_COLUMN_NAME
    try:
        SCORE_COLUMN_NAME = job["jobParameters"]["SCORE_COLUMN_NAME"]
        logger.info(f"Loaded SCORE_COLUMN_NAME from job parameters: '{SCORE_COLUMN_NAME}'")
    except Exception:
        SCORE_COLUMN_NAME = 'score'
        logger.info(f"Using default SCORE_COLUMN_NAME: '{SCORE_COLUMN_NAME}'")

    # Parameter: SCORE_FALSE_VALUE
    try:
        SCORE_FALSE_VALUE = job["jobParameters"]["SCORE_FALSE_VALUE"]
        logger.info(f"Loaded SCORE_FALSE_VALUE from job parameters: '{SCORE_FALSE_VALUE}'")
    except Exception:
        SCORE_FALSE_VALUE = 'NO'
        logger.info(f"Using default SCORE_FALSE_VALUE: '{SCORE_FALSE_VALUE}'")

    # Parameter: SCORE_TRUE_VALUE
    try:
        SCORE_TRUE_VALUE = job["jobParameters"]["SCORE_TRUE_VALUE"]
        logger.info(f"Loaded SCORE_TRUE_VALUE from job parameters: '{SCORE_TRUE_VALUE}'")
    except Exception:
        SCORE_TRUE_VALUE = 'YES'
        logger.info(f"Using default SCORE_TRUE_VALUE: '{SCORE_TRUE_VALUE}'")

    # --- Calculation Parameters ---

    # Parameter: METRICS_TO_CALC
    try:
        METRICS_TO_CALC = job["jobParameters"]["METRICS_TO_CALC"]
        logger.info(f"Loaded METRICS_TO_CALC from job parameters: {METRICS_TO_CALC}")
    except Exception:
        METRICS_TO_CALC = ['SEN', 'SP']
        logger.info(f"Using default METRICS_TO_CALC: {METRICS_TO_CALC}")

    # Parameter: BINS_TO_CALC
    try:
        BINS_TO_CALC = job["jobParameters"]["BINS_TO_CALC"]
        logger.info(f"Loaded BINS_TO_CALC from job parameters: {BINS_TO_CALC}")
    except Exception:
        BINS_TO_CALC = ['W', 'MS', 'YS'] # Weekly, Month-Start, Year-Start
        logger.info(f"Using default BINS_TO_CALC: {BINS_TO_CALC}")
        
    # Parameter: NUMERIC_AGGS_TO_CALC
    try:
        NUMERIC_AGGS_TO_CALC = job["jobParameters"]["NUMERIC_AGGS_TO_CALC"]
        if NUMERIC_AGGS_TO_CALC is None:
            # Handle case where parameter is explicitly set to null
            NUMERIC_AGGS_TO_CALC = {} 
            logger.info("NUMERIC_AGGS_TO_CALC set to 'null', using {}.")
        else:
            logger.info(f"Loaded NUMERIC_AGGS_TO_CALC from job parameters: {NUMERIC_AGGS_TO_CALC}")
    except Exception:
        NUMERIC_AGGS_TO_CALC = {'patient_age': 'mean'}
        logger.info(f"Using default NUMERIC_AGGS_TO_CALC: {NUMERIC_AGGS_TO_CALC}")

    logger.info("Initialization complete.")


def _format_df_for_timeline_graph(df, title, date_format='%Y-%m-%dT%H:%M:%S'):
    """
    Helper function to convert a binned metrics DataFrame into the
    required JSON timeline graph format for the ModelOp UI.

    Args:
        df (pd.DataFrame): The DataFrame of binned metrics (index=timestamp, cols=metrics).
        title (str): The title for the graph.
        date_format (str, optional): The strftime format for the timestamp.

    Returns:
        dict: A JSON-serializable dictionary formatted for the timeline graph.
    """
    
    # Base structure of the JSON object ModelOp expects
    graph_json = {
        "title": title,
        "x_axis_label": TIMESTAMP_COLUMN, # Use global var for consistency
        "y_axis_label": "Metric Value",
        "data": {} # Data will be populated below
    }
    
    # Use the specified date format string
    iso_format = date_format

    # Iterate over each *column* in the DataFrame (e.g., 'SEN', 'SP', 'patient_age')
    for metric_name in df.columns:
        metric_data = [] # This will hold the [timestamp, value] pairs
        
        # Iterate over each *row* (timestamp) for that metric
        # .items() gives us (index, value) pairs, where index is the timestamp
        for timestamp, value in df[metric_name].items():
            
            # Only include non-null values in the graph data
            if pd.notna(value):
                # Format: [timestamp_string, numeric_value]
                metric_data.append([timestamp.strftime(iso_format), value])
        
        # Add this metric's timeseries data to the 'data' dictionary
        # The key is the metric name (which becomes the legend label)
        graph_json["data"][metric_name] = metric_data
        
    return graph_json


# modelop.metrics
def metrics(baseline: pd.DataFrame):
    """
    Calculates binned (Weekly, Monthly, Yearly) performance metrics
    and/or numeric aggregations based on global variables set in init().
    
    This is the main entry point function called by ModelOp to run the monitoring job.

    Args:
        baseline (pd.DataFrame): The input data (e.g., from a 'baseline' dataset)
                                 passed by the ModelOp engine.
    
    Yields:
        dict: A dictionary of metrics, including the formatted JSON
              for the timeline graphs.
    """

    # Use the global logger configured in init()
    logger.info("Starting metrics function execution.")

    # --- Input Data Check ---
    if baseline is None or baseline.empty:
        logger.warning("Input DataFrame is empty. Yielding empty results.")
        # Yield an empty structure to prevent errors in the UI
        yield {
            "firstPredictionDate":'',
            "lastPredictionDate":'',
            "baseline_time_line_graph_weekly": {"title": "Weekly Metrics", "data": {}},
            "baseline_time_line_graph_monthly": {"title": "Monthly Metrics", "data": {}},
            "baseline_time_line_graph_yearly": {"title": "Yearly Metrics", "data": {}}
        }
        return # Stop execution

    ### --- READ GLOBALS AND CALL BINNED METRICS FUNCTION --- ###
    
    # 1. Call the universal binned metrics function
    # All parameters are read from the global scope set in init()
    # We pass the global variables directly to the function,
    # no longer needing to build dictionary maps.
    logger.info("Calculating binned performance metrics...")
    binned_results = calculate_binned_metrics(
        df=baseline,
        timestamp_col=TIMESTAMP_COLUMN,
        bins=BINS_TO_CALC,
        label_col=LABEL_COLUMN_NAME,
        label_true=LABEL_TRUE_VALUE,
        label_false=LABEL_FALSE_VALUE,
        score_col=SCORE_COLUMN_NAME,
        score_true=SCORE_TRUE_VALUE,
        score_false=SCORE_FALSE_VALUE,
        metrics=METRICS_TO_CALC,
        numeric_aggregations=NUMERIC_AGGS_TO_CALC,
        logger=logger
    )

    # --- Calculate first and last prediction dates ---
    try:
        logger.info("Calculating first and last prediction dates.")
        first_prediction_date = str(pd.to_datetime(baseline[TIMESTAMP_COLUMN]).min().date())
        last_prediction_date = str(pd.to_datetime(baseline[TIMESTAMP_COLUMN]).max().date())
    except Exception as e:
        logger.warning(f"Could not parse dates from TIMESTAMP_COLUMN '{TIMESTAMP_COLUMN}'. Setting to empty string. Error: {e}")
        first_prediction_date = ""
        last_prediction_date = ""
   
    # --- 3. Format results into the required JSON graph output ---
    logger.info("Formatting results for JSON output...")
    
    # Determine columns for a default empty DataFrame.
    # This prevents errors if a bin (e.g., 'W') has no data.
    default_cols = []
    if METRICS_TO_CALC:
        default_cols.extend(METRICS_TO_CALC)
    if NUMERIC_AGGS_TO_CALC:
        default_cols.extend(NUMERIC_AGGS_TO_CALC.keys())
    if not default_cols: # Failsafe
        default_cols = None

    # Use .get() to safely retrieve the DataFrame for each bin.
    # If the key (e.g., 'W') doesn't exist, it uses the default empty DataFrame.
    
    # Get the title for graphs. Can be customized.
    metrics_str = " & ".join(METRICS_TO_CALC) if METRICS_TO_CALC else "Metrics"
    aggs_str = " & ".join(NUMERIC_AGGS_TO_CALC.keys()) if NUMERIC_AGGS_TO_CALC else ""
    base_title = f"{metrics_str}{' & ' if METRICS_TO_CALC and NUMERIC_AGGS_TO_CALC else ''}{aggs_str}"

    weekly_graph = _format_df_for_timeline_graph(
        df = binned_results.get('W', pd.DataFrame(columns=default_cols)),
        title = f"Weekly {base_title}",
        date_format='%Y-%m-%d' # Use simple date format for weekly
    )
    
    monthly_graph = _format_df_for_timeline_graph(
        df = binned_results.get('MS', pd.DataFrame(columns=default_cols)),
        title = f"Monthly {base_title}",
        date_format='%Y-%m-%d' # Use simple date format for monthly
    )
    
    yearly_graph = _format_df_for_timeline_graph(
        df = binned_results.get('YS', pd.DataFrame(columns=default_cols)),
        title = f"Yearly {base_title}",
        date_format='%Y-%m-%d' # Use simple date format for yearly
    )

    # --- 4. Yield the final JSON-like dict ---
    # This is the final output of the ModelOp job
    logger.info("Yielding final metrics dictionary.")
    yield {
        "firstPredictionDate": first_prediction_date,
        "lastPredictionDate": last_prediction_date,
        "baseline_time_line_graph_weekly": weekly_graph,
        "baseline_time_line_graph_monthly": monthly_graph,
        "baseline_time_line_graph_yearly": yearly_graph
    }

#
# This main method is utilized to simulate what the engine will do when
# calling the above ModelOp functions. It is used for local development
# and testing.
#
def main():
    """
    Local testing function.
    
    1. Simulates the 'init' call with a sample parameter string.
    2. Reads a local CSV file ('baseline_df').
    3. Calls the 'metrics' function with the local data.
    4. Prints the JSON output and saves it to a file.
    """
    
    # Create a dummy init_param string.
    # In a real job, ModelOp Center provides this.
    # You can modify this JSON to test different job parameters.
    # Note: We use 'null' for NUMERIC_AGGS_TO_CALC to test that logic.
    # We also override the default 'label' to 'actual_label'
    test_params = {
        "rawJson": json.dumps({
            "jobParameters": {
                "TIMESTAMP_COLUMN": "Date",
                "LABEL_COLUMN_NAME": "label", # <-- IMPORTANT: Change 'label' to match your CSV
                "LABEL_FALSE_VALUE": False,
                "LABEL_TRUE_VALUE": True,
                "SCORE_COLUMN_NAME": "score", # <-- IMPORTANT: Change 'score' to match your CSV
                "SCORE_FALSE_VALUE": "NO",
                "SCORE_TRUE_VALUE": "YES",
                "METRICS_TO_CALC": ["SEN", "SP", "ACC", "F1"],
                "BINS_TO_CALC": ["W", "MS"],
                "NUMERIC_AGGS_TO_CALC": {"patient_age": "mean"} # <-- IMPORTANT: Change 'patient_age' to match your CSV
            }
        })
    }
    
    # 1. initialize global variables
    print("--- Calling init() ---")
    init(test_params)
    print("--- init() complete ---")
    
    # 2. Read local data
    data_file = 'synthetic_2021_2024_prediction_data.csv'
    try:
        baseline_df = pd.read_csv(data_file)
        print(f"\nSuccessfully read data from '{data_file}'")
    except FileNotFoundError:
        print(f"Error: '{data_file}' not found.")
        print("Please ensure the synthetic data file is in the correct directory to run main().")
        return
    except Exception as e:
        print(f"Error reading '{data_file}': {e}")
        return

    # 3. Call the metrics function and get the result
    # 'metrics' is a generator, so 'next()' gets the first yielded value
    print("--- Calling metrics() ---")
    result = next(metrics(baseline_df))
    print("--- metrics() complete ---")
    
    # 4. Write output to a local JSON file
    output_filename = 'universal_performance_2021_2024_metrics_example_output.json'
    with open(output_filename, 'w') as f:
        # We wrap 'result' in a list to match ModelOp's expected output shape
        json.dump([result], f, indent=4)
        
    print(f"\nOutput written to '{output_filename}'")
    
    # Optionally, print the 'data' part of the weekly graph for a quick check
    # print("\n--- Sample of Weekly Graph Data ---")
    # print(json.dumps(result.get("baseline_time_line_graph_weekly", {}).get("data", {}), indent=2))


if __name__ == '__main__':
    main()