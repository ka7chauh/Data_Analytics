import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve, auc
from statsmodels.stats.weightstats import ttest_ind
from typing import Optional, List, Union
from datetime import datetime



def check_missing_values(df, feature_name):
    """
    Checks the percentage of missing values for a given feature in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        feature_name (str): The name of the feature to check.

    Returns:
        float: The percentage of missing values (between 0 and 1).
    """
    missing_count = df[feature_name].isnull().sum()
    total_count = len(df)
    return missing_count / total_count if total_count > 0 else 0

def check_zero_values(df, feature_name):
    """
    Checks the percentage of zero values for a given feature in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        feature_name (str): The name of the feature to check.

    Returns:
        float: The percentage of zero values (between 0 and 1).
    """
    zero_count = (df[feature_name] == 0).sum()
    total_count = len(df)
    return zero_count / total_count if total_count > 0 else 0


def check_outliers(df, feature_name):
    """
    Checks the percentage of outliers for a given feature in a DataFrame using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        feature_name (str): The name of the feature to check.

    Returns:
        float: The percentage of outliers (between 0 and 1).
    """
    Q1 = df[feature_name].quantile(0.25)
    Q3 = df[feature_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # upper_bound = Q3
    outlier_count = ( (df[feature_name] < lower_bound) |(df[feature_name] > upper_bound)).sum()
    total_count = len(df)
    return outlier_count / total_count if total_count > 0 else 0

def calculate_psi(current, reference, bins=10):
    """
    Calculates the Population Stability Index (PSI) for a given variable.

    Args:
        current (pd.Series): The current distribution of the variable.
        reference (pd.Series): The reference distribution of the variable.
        bins (int): The number of bins to use for calculating PSI.

    Returns:
        float: The PSI value.
    """
    # Ensure no empty bins, and handle edge case of zero counts.
    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator > 0 else 0

    bins = np.linspace(min(current.min(), reference.min()), max(current.max(), reference.max()), bins + 1)
    reference_counts, _ = np.histogram(reference, bins=bins)
    current_counts, _ = np.histogram(current, bins=bins)

    total_reference = len(reference)
    total_current = len(current)

    psi_values = []
    for i in range(len(reference_counts)):
        ref_pct = safe_divide(reference_counts[i], total_reference)
        curr_pct = safe_divide(current_counts[i], total_current)
        if ref_pct > 0 and curr_pct > 0:
            psi_values.append((ref_pct - curr_pct) * log(ref_pct / curr_pct))
    return sum(psi_values)


def calculate_model_psi(current_predictions, reference_predictions, bins=20):
    """
    Calculates the Population Stability Index (PSI) for model scores.

    Args:
        current_predictions (pd.Series): Predicted scores for the current data.
        reference_predictions (pd.Series): Predicted scores for the reference data.
        bins (int): Number of bins for PSI calculation.

    Returns:
        float: The PSI value.
    """
    return calculate_psi(current_predictions, reference_predictions, bins)


def calculate_gini(data, score_col, target_col, date_col, group_col, reference_quarter=None):
    """
    Calculates Gini coefficient for each group over time, using a specific quarter as reference if provided

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        score_col (str): Name of the score column.
        target_col (str): Name of the target column.
        date_col (str): Name of the date column.
        group_col (str): Name of the grouping column.
        reference_quarter (str, optional): Quarter to use as reference. If None, uses earliest.

    Returns:
        pd.DataFrame: Gini coefficient by group and date.
    """
    grouped_data = data.groupby([date_col, group_col])
    gini_values = grouped_data.apply(lambda x: 2 * roc_auc_score(x[target_col], x[score_col]) - 1).reset_index(name='Gini')

    if reference_quarter:
        ref_data = data[data[date_col] == reference_quarter]
        if len(ref_data) > 0:  # check if ref quarter exists
            ref_gini = ref_data.groupby(group_col).apply(lambda x: 2 * roc_auc_score(x[target_col], x[score_col]) - 1).reset_index(name='Reference_Gini')
            gini_values = gini_values.merge(ref_gini, on=group_col, how='left')
            gini_values['Gini_Change'] = gini_values['Gini'] - gini_values['Reference_Gini']
        else:
            gini_values['Gini_Change'] = None
    else:
        # Use the earliest period as the reference
        earliest_period = gini_values[date_col].min()
        ref_gini = gini_values[gini_values[date_col] == earliest_period].rename(columns={'Gini': 'Reference_Gini'}).drop(columns=[date_col])
        gini_values = gini_values.merge(ref_gini, on=group_col, how='left')
        gini_values['Gini_Change'] = gini_values['Gini'] - gini_values['Reference_Gini']
    return gini_values


def calculate_psi_group(data, score_col, date_col, group_col, reference_quarter=None):
    """
    Calculates PSI for each group over time, using a specific quarter as reference if provided.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        score_col (str): Name of the score column.
        date_col (str): Name of the date column.
        group_col (str): Name of the grouping column.
        reference_quarter (str, optional): Quarter to use as reference. If None, uses earliest.

    Returns:
        pd.DataFrame: PSI by group and date.
    """
    grouped_data = data.groupby([date_col, group_col])
    psi_values = grouped_data.apply(lambda x: calculate_psi(x[score_col], data[data[date_col] == x[date_col].iloc[0]][score_col])).reset_index(name='PSI')

    if reference_quarter:
        ref_data = data[data[date_col] == reference_quarter]
        if len(ref_data) > 0:
            ref_scores = ref_data.groupby(group_col)[score_col].apply(lambda x: x).reset_index(name='Reference_Scores')
            psi_values = psi_values.merge(ref_scores, on=group_col, how='left')
            psi_values['PSI'] = psi_values.apply(lambda row: calculate_psi(data[data[date_col] == row[date_col]][score_col], row['Reference_Scores']), axis=1)
            psi_values = psi_values.drop(columns=['Reference_Scores'])

    else:
        # Use the earliest period as the reference
        earliest_period = psi_values[date_col].min()
        ref_scores = data[data[date_col] == earliest_period].groupby(group_col)[score_col].apply(lambda x: x).reset_index(name='Reference_Scores')
        psi_values = psi_values.merge(ref_scores, on=group_col, how='left')
        psi_values['PSI'] = psi_values.apply(lambda row: calculate_psi(data[data[date_col] == row[date_col]][score_col], row['Reference_Scores']), axis=1)
        psi_values = psi_values.drop(columns=['Reference_Scores'])
    return psi_values


import pandas as pd, numpy as np, os, re, math, time
from tqdm import tqdm
def is_monotonic(temp_series):
    return all(temp_series[i] <= temp_series[i + 1] for i in range(len(temp_series) - 1)) or all(temp_series[i] >= temp_series[i + 1] for i in range(len(temp_series) - 1))

def prepare_bins(bin_data, c_i, target_col, max_bins):
    force_bin = True
    binned = False
    remarks = np.nan
    # ----------------- Monotonic binning -----------------
    for n_bins in range(max_bins, 2, -1):
        try:
            bin_data[c_i + "_bins"] = pd.qcut(bin_data[c_i], n_bins, duplicates="drop")
            monotonic_series = bin_data.groupby(c_i + "_bins")[target_col].mean().reset_index(drop=True)
            if is_monotonic(monotonic_series):
                force_bin = False
                binned = True
                remarks = "binned monotonically"
                break
        except:
            pass
    # ----------------- Force binning -----------------
    # creating 2 bins forcefully because 2 bins will always be monotonic
    if force_bin or (c_i + "_bins" in bin_data and bin_data[c_i + "_bins"].nunique() < 2):
        _min=bin_data[c_i].min()
        _mean=bin_data[c_i].mean()
        _max=bin_data[c_i].max()
        bin_data[c_i + "_bins"] = pd.cut(bin_data[c_i], [_min, _mean, _max], include_lowest=True)
        if bin_data[c_i + "_bins"].nunique() == 2:
            binned = True
            remarks = "binned forcefully"
    
    if binned:
        return c_i + "_bins", remarks, bin_data[[c_i, c_i+"_bins", target_col]].copy()
    else:
        remarks = "couldn't bin"
        return c_i, remarks, bin_data[[c_i, target_col]].copy()

# calculate WOE and IV for every group/bin/class for a provided feature
def iv_woe_4iter(binned_data, target_col, class_col):
    if "_bins" in class_col:
        binned_data[class_col] = binned_data[class_col].cat.add_categories(['Missing'])
        binned_data[class_col] = binned_data[class_col].fillna("Missing")
        temp_groupby = binned_data.groupby(class_col).agg({class_col.replace("_bins", ""):["min", "max"],
                                                           target_col: ["count", "sum", "mean"]}).reset_index()
    else:
        binned_data[class_col] = binned_data[class_col].fillna("Missing")
        temp_groupby = binned_data.groupby(class_col).agg({class_col:["first", "first"],
                                                           target_col: ["count", "sum", "mean"]}).reset_index()  
    
    temp_groupby.columns = ["sample_class", "min_value", "max_value", "sample_count", "event_count", "event_rate"]
    temp_groupby["non_event_count"] = temp_groupby["sample_count"] - temp_groupby["event_count"]
    temp_groupby["non_event_rate"] = 1 - temp_groupby["event_rate"]
    temp_groupby = temp_groupby[["sample_class", "min_value", "max_value", "sample_count",
                                 "non_event_count", "non_event_rate", "event_count", "event_rate"]]
    
    if "_bins" not in class_col and "Missing" in temp_groupby["min_value"]:
        temp_groupby["min_value"] = temp_groupby["min_value"].replace({"Missing": np.nan})
        temp_groupby["max_value"] = temp_groupby["max_value"].replace({"Missing": np.nan})
    temp_groupby["feature"] = class_col
    if "_bins" in class_col:
        temp_groupby["sample_class_label"]=temp_groupby["sample_class"].replace({"Missing": np.nan}).astype('category').cat.codes.replace({-1: np.nan})
    else:
        temp_groupby["sample_class_label"]=np.nan
    temp_groupby = temp_groupby[["feature", "sample_class", "sample_class_label", "sample_count", "min_value", "max_value",
                                 "non_event_count", "non_event_rate", "event_count", "event_rate"]]
    
    """
    **********get distribution of good and bad
    """
    temp_groupby['distbn_non_event'] = temp_groupby["non_event_count"]/temp_groupby["non_event_count"].sum()
    temp_groupby['distbn_event'] = temp_groupby["event_count"]/temp_groupby["event_count"].sum()

    temp_groupby['woe'] = np.log(temp_groupby['distbn_non_event'] / temp_groupby['distbn_event'])
    temp_groupby['iv'] = (temp_groupby['distbn_non_event'] - temp_groupby['distbn_event']) * temp_groupby['woe']
    
    temp_groupby["woe"] = temp_groupby["woe"].replace([np.inf,-np.inf],0)
    temp_groupby["iv"] = temp_groupby["iv"].replace([np.inf,-np.inf],0)
    
    return temp_groupby

"""
- iterate over all features.
- calculate WOE & IV for there classes.
- append to one DataFrame woe_iv.
"""
def var_iter(data, target_col, max_bins):
    woe_iv = pd.DataFrame()
    remarks_list = []

    tqdm.pandas(desc="Processing Features for IV & WoE")

    print(f"Processing {len(data.columns) - 1} features...\n")

    for c_i in tqdm(data.columns, desc="Feature Processing"):
        if c_i != target_col:
            c_i_start_time = time.time()
            remarks = ""
            
            # Check if binning is required
            if np.issubdtype(data[c_i], np.number) and data[c_i].nunique() > 2:
                class_col, remarks, binned_data = prepare_bins(data[[c_i, target_col]].copy(), c_i, target_col, max_bins)
                agg_data = iv_woe_4iter(binned_data.copy(), target_col, class_col)
            else:
                agg_data = iv_woe_4iter(data[[c_i, target_col]].copy(), target_col, c_i)
                remarks = "categorical"

            # Store remarks for tracking
            remarks_list.append({"feature": c_i, "remarks": remarks})

            # Append results
            woe_iv = pd.concat([woe_iv, agg_data], ignore_index=True)

            # Print time taken for each feature
            # print(f"✅ Processed {c_i} in {round(time.time() - c_i_start_time, 2)} sec. Remarks: {remarks}")

    # print("\n🚀 IV & WoE Calculation Completed!\n")
    return woe_iv, pd.DataFrame(remarks_list)


# after getting woe and iv for all classes of features calculate aggregated IV values for features.
def get_iv_woe(data, target_col, max_bins):
    func_start_time = time.time()
    woe_iv, binning_remarks = var_iter(data, target_col, max_bins)
    print("------------------IV and WOE calculated for individual groups.------------------")
    print("Total time elapsed: {} minutes".format(round((time.time() - func_start_time) / 60, 3)))
    
    woe_iv["feature"] = woe_iv["feature"].replace("_bins", "", regex=True)    
    woe_iv = woe_iv[["feature", "sample_class", "sample_class_label", "sample_count", "min_value", "max_value",
                     "non_event_count", "non_event_rate", "event_count", "event_rate", 'distbn_non_event',
                     'distbn_event', 'woe', 'iv']]
    
    iv = woe_iv.groupby("feature")[["iv"]].agg(["sum", "count"]).reset_index()
    print("------------------Aggregated IV values for features calculated.------------------")
    print("Total time elapsed: {} minutes".format(round((time.time() - func_start_time) / 60, 3)))
    
    iv.columns = ["feature", "iv", "number_of_classes"]
    null_percent_data=pd.DataFrame(data.isnull().mean()).reset_index()
    null_percent_data.columns=["feature", "feature_null_percent"]
    iv=iv.merge(null_percent_data, on="feature", how="left")
    print("------------------Null percent calculated in features.------------------")
    print("Total time elapsed: {} minutes".format(round((time.time() - func_start_time) / 60, 3)))
    iv = iv.merge(binning_remarks, on="feature", how="left")
    woe_iv = woe_iv.merge(iv[["feature", "iv", "remarks"]].rename(columns={"iv": "iv_sum"}), on="feature", how="left")
    print("------------------Binning remarks added and process is complete.------------------")
    print("Total time elapsed: {} minutes".format(round((time.time() - func_start_time) / 60, 3)))
    return iv, woe_iv.replace({"Missing": np.nan})

def categorize_quality(score):
    if score >= 90:
        return "Good"
    elif score >= 70:
        return "Warning"
    else:
        return "Poor"
    

    
def print_psi_analysis(psi_result):
    """Prints and formats the PSI results.

       Args:
           psi_result (pd.DataFrame): DataFrame containing the PSI values.
    """
    print(psi_result.round(3).to_markdown(index=False))
    
def analyze_psi_bins(current_data, reference_data, score_col, bins=10, psi_threshold=0.1, bin_psi_threshold=0.02):
    """Analyzes PSI within each bin of a score distribution and identifies significant shifts.

    Args:
        current_data (pd.DataFrame): DataFrame containing current data.
        reference_data (pd.DataFrame): DataFrame containing reference data.
        score_col (str): Name of the score column.
        bins (int): Number of bins.
        psi_threshold (float): Overall PSI threshold for the feature.
        bin_psi_threshold (float): PSI threshold for individual bins.

    Returns:
        pd.DataFrame: Analysis results including bin ranges, PSI, and status.
    """
    combined_data = pd.concat([
        current_data[[score_col]].assign(dataset='Current'),
        reference_data[[score_col]].assign(dataset='Reference')
    ])

    min_val = combined_data[score_col].min()
    max_val = combined_data[score_col].max()
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    bin_labels = [f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})" for i in range(bins)]

    combined_data['bin'] = pd.cut(combined_data[score_col], bins=bin_edges, labels=bin_labels, include_lowest=True)

    reference_counts = combined_data[combined_data['dataset'] == 'Reference']['bin'].value_counts().reindex(bin_labels, fill_value=0)
    current_counts = combined_data[combined_data['dataset'] == 'Current']['bin'].value_counts().reindex(bin_labels, fill_value=0)

    total_reference = reference_counts.sum()
    total_current = current_counts.sum()

    results = pd.DataFrame({
        'Bin': bin_labels,
        'Reference_Count': reference_counts,
        'Current_Count': current_counts,
        'Reference_Pct': (reference_counts / total_reference).fillna(0),
        'Current_Pct': (current_counts / total_current).fillna(0)
    })

def calculate_performance_metrics(data, target, features):
    """Calculates Information Value (IV), correlation, KS statistic, and Gini for given features.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        target (pd.Series): Target variable.
        features (list): List of features to analyze.

    Returns:
        pd.DataFrame: DataFrame with calculated metrics.
    """
    metrics = []
    for feature in features:
        if not pd.api.types.is_numeric_dtype(data[feature]):
            continue  # Skip non-numeric features for these calculations

        # Calculate Information Value (IV)
        def calculate_iv(df, feature, target):
            df['target_0'] = 1 - df[target]
            grouped = df.groupby(feature).agg({'target_0': 'sum', target: 'sum'}).reset_index()
            grouped['dist_0'] = grouped['target_0'] / grouped['target_0'].sum()
            grouped['dist_1'] = grouped[target] / grouped[target].sum()
            grouped['woe'] = np.log(grouped['dist_1'] / grouped['dist_0'])
            grouped['iv'] = (grouped['dist_1'] - grouped['dist_0']) * grouped['woe']
            return grouped['iv'].sum()

        iv = calculate_iv(data[[feature, target]].copy(), feature, target)

        # Calculate correlation
        correlation = data[[feature, target]].corr().iloc[0, 1]

        # Calculate KS statistic
        ks_statistic = ks_2samp(data[data[target == 0][feature]], data[data[target == 1][feature]]).statistic

        # Calculate Gini
        gini = 2 * roc_auc_score(target, data[feature]) - 1

        metrics.append({
            'Feature': feature,
            'IV': iv,
            'Correlation': correlation,
            'KS_Statistic': ks_statistic,
            'Gini': gini
        })
    return pd.DataFrame(metrics)


from sklearn.metrics import roc_auc_score

def calculate_auc_gini(model, X_test, y_test):
    """
    Calculates the Area Under the Receiver Operating Characteristic curve (AUC)
    and the Gini coefficient for a trained binary classification model.

    Args:
        model: A trained binary classification model with a predict_proba method.
        X_test: The test set features.
        y_test: The true labels for the test set.

    Returns:
        A tuple containing the AUC score and the Gini coefficient.
    """
    try:
        # Get probability predictions for the positive class
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate AUC
        auc_score = roc_auc_score(y_test, y_pred_proba)

        # Calculate Gini coefficient
        gini_coefficient = 2 * auc_score - 1

        return auc_score, gini_coefficient

    except AttributeError:
        print("Error: The provided model does not have a 'predict_proba' method.")
        return None, None
    except ValueError:
        print("Error: The number of classes in y_true is not equal to the number of columns in y_score.")
        return None, None
    

from sklearn.metrics import roc_auc_score, roc_curve

def generate_decile_table(model, X_test, y_test,cut):
    """
    Generate decile table, Gini coefficient, and KS statistic for a given model.
    
    Parameters:
    - model: Trained classification model with `predict_proba` method
    - X_test: Test features
    - y_test: Actual labels (binary: 0/1)
    
    Returns:
    - decile_table (pd.DataFrame)
    - gini (float)
    - ks (float)
    """
    

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    

    df = pd.DataFrame({'y_true': y_test, 'y_pred_proba': y_pred_proba})
    
 
    df['decile'] = pd.qcut(df['y_pred_proba'],cut, labels=False, duplicates='drop')
    df['decile'] = cut - df['decile']  # highest score = decile 0

    # Aggregate by decile
    decile_table = df.groupby('decile').agg(
        count=('y_true', 'count'),
        events=('y_true', 'sum'),
        non_events=('y_true', lambda x: x.count() - x.sum()),
        min_score=('y_pred_proba', 'min'),
        max_score=('y_pred_proba', 'max'),
        avg_score=('y_pred_proba', 'mean'),
    ).reset_index()

    # Add cumulative and event rate columns
    decile_table['cum_events'] = decile_table['events'].cumsum()
    decile_table['cum_non_events'] = decile_table['non_events'].cumsum()
    decile_table['cum_total'] = decile_table['count'].cumsum()
    decile_table['event_rate'] = decile_table['events'] / decile_table['count']
    
    # Gini = 2*AUC - 1
    auc = roc_auc_score(y_test, y_pred_proba)
    gini = 2 * auc - 1

    # KS Statistic
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ks = max(tpr - fpr)

    return decile_table, gini, ks



from tqdm import tqdm
import numpy as np
import pandas as pd

def drop_uninformative_columns(df, constant_threshold=2, near_zero_threshold=0.50, missing_threshold=0.55):
    """
    Drops constant, near-zero variance, high-missing, and duplicate columns from a DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    constant_threshold (int): Minimum unique values a column must have to be retained (default = 2)
    near_zero_threshold (float): Maximum proportion of the most frequent value in a column (default = 0.99)
    missing_threshold (float): Maximum proportion of missing values allowed per column (default = 0.75)

    Returns:
    pd.DataFrame: Cleaned DataFrame with uninformative columns removed
    """

    print("🔍 Dropping uninformative columns...")

    # 1. Drop constant columns
    nunique = df.nunique(dropna=False)
    mask_constant = nunique > constant_threshold
    df_cleaned = df.loc[:, mask_constant]

    print(f"Step 1 ✅ Dropped constant columns: {df.shape[1] - df_cleaned.shape[1]}")

    # 2. Drop near-zero variance columns
    def max_freq_ratio(col):
        values, counts = np.unique(col[~pd.isnull(col)], return_counts=True)
        if counts.size == 0:
            return 1.0  # all values were NaN
        return counts.max() / counts.sum()

    print("Step 2 🔄 Evaluating near-zero variance columns...")
    tqdm.pandas(desc="Computing freq ratios")
    freq_ratios = df_cleaned.progress_apply(max_freq_ratio, axis=0)
    mask_nzv = freq_ratios < near_zero_threshold
    df_cleaned = df_cleaned.loc[:, mask_nzv]

    print(f"Step 2 ✅ Dropped near-zero variance columns: {mask_constant.sum() - mask_nzv.sum()}")

    # 3. Drop high-missing-value columns
    missing_ratio = df_cleaned.isnull().mean()
    mask_missing = missing_ratio < missing_threshold
    df_cleaned = df_cleaned.loc[:, mask_missing]

    print(f"Step 3 ✅ Dropped high-missing-value columns: {mask_nzv.sum() - mask_missing.sum()}")

#     # 4. Drop duplicate columns
#     df_cleaned_T = df_cleaned.T
#     df_cleaned = df_cleaned_T.drop_duplicates().T

#     print(f"Step 4 ✅ Dropped duplicate columns")

    print(f"\n📊 Final columns: {df_cleaned.shape[1]} (Dropped {df.shape[1] - df_cleaned.shape[1]} total columns)")

    return df_cleaned




def calculate_psi(current, reference, bins=10):
    """
    Calculates the Population Stability Index (PSI) for a given variable,
    including a separate bin for NaN values.
    """
    import numpy as np
    import pandas as pd

    if isinstance(current, pd.DataFrame):
        raise ValueError("Expected 'current' to be a Series, got DataFrame.")
    if isinstance(reference, pd.DataFrame):
        raise ValueError("Expected 'reference' to be a Series, got DataFrame.")

    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator > 0 else 0

    # Separate NaNs
    current_notna = current[~current.isna()]
    reference_notna = reference[~reference.isna()]
    current_na_count = current.isna().sum()
    reference_na_count = reference.isna().sum()

    # Bin non-NaN values
    min_val = min(current_notna.min(), reference_notna.min())
    max_val = max(current_notna.max(), reference_notna.max())

    bins_edges = np.linspace(min_val, max_val, bins + 1)

    reference_counts, _ = np.histogram(reference_notna, bins=bins_edges)
    current_counts, _ = np.histogram(current_notna, bins=bins_edges)

    # Append NaN bin
    reference_counts = np.append(reference_counts, reference_na_count)
    current_counts = np.append(current_counts, current_na_count)

    total_reference = len(reference)
    total_current = len(current)

    psi_values = []
    for i in range(len(reference_counts)):
        ref_pct = safe_divide(reference_counts[i], total_reference)
        curr_pct = safe_divide(current_counts[i], total_current)
        if ref_pct > 0 and curr_pct > 0:
            psi_values.append((ref_pct - curr_pct) * np.log(ref_pct / curr_pct))

    return sum(psi_values)




def calculate_dataframe_psi(df_current,df_reference):
    if sorted(df_current.columns) != sorted(df_reference.columns):
         raise ValueError("features are not same")
    features = df_current.columns
    psi_list = []
    for ff in features:
        current,reference = df_current[ff],df_reference[ff]
        psi = calculate_psi(current,reference)
        psi_list.append({'feature': ff, 'psi': psi.round(4)})
        psi_df = pd.DataFrame(psi_list)
    return psi_df.sort_values(by='psi', ascending=False)
        
    