import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

# class BallData:
#     def __init__(self):
#         self.data = None
#         self.datadir = None
#         self.subject = None
#         self.task = None
#         self.subs = None

# TODO: TURN INTO EFFICIENT CLASS, AT SOME POINT, BUT NOW TOO EARLY

def get_data(subject:str | None=None, datadir: str | None=None, task: str | None = "ball_hue", session_specific:bool=False, verbose:bool=False):
    datadir = "/Users/wiegerscheurer/repos/physicspred/data" if datadir is None else datadir
    
    subs = [sub for sub in os.listdir(datadir) if sub.startswith("sub")] if subject is None else [subject]
        

    file_stack = []
    for sub in subs:

        datafiles = os.listdir(f"{datadir}/{sub}/{task}/")
        # # Check if there is a full_experiment file
        if "full_experiment.csv" in datafiles and not session_specific:
            print(f"Loading full_experiment file for {sub}") if verbose else None
            this_file = pd.read_csv(f"{datadir}/{sub}/{task}/full_experiment.csv")
            file_stack.append(this_file)
        else:
            # Loop over session-specific files
            for file in datafiles:
                if file.endswith(".csv") and not file.startswith("design_matrix") and not file.startswith("full_experiment"):
                    this_file = pd.read_csv(f"{datadir}/{sub}/{task}/{file}")
                    file_stack.append(this_file)

        # Check if there's a column with "ball_speed", and if not, create one and fill with None
        if "ball_speed" not in this_file.columns:
            this_file["ball_speed"] = None
            
        # Concatenate all DataFrames in the list into a single DataFrame
        combined_df = pd.concat(file_stack, ignore_index=True)

    return combined_df

def get_false_positives(df):
    return len(df[(df["ball_change"] == False) & (df["response"].notnull())])

def get_false_negatives(df):
    return len(df[(df["ball_change"] == True) & (df["response"].isna())])

def get_true_positives(df):
    return len(df[(df["ball_change"] == True) & (df["accuracy"] == True)])

def get_true_negatives(df):
    return len(df[(df["ball_change"] == 0) & (df["response"].isna())])

# This is the same as hit rate, but now I do it a shit load more efficiently
def get_sensitivity(df, hypothesis:str = "both", include_dubtrials:bool | str=False, return_df:bool=False):
    """Precision: True positives / (True positives + False negatives)
    Args:
        df (pd.dataframe): The data
        hypothesis (str): The hypothesis to test. Can be either "simulation", "abstraction" or "both"
        include_dubtrials (bool): Whether to include trials where both hypotheses are congruent
        return_df (bool): Whether to return the filtered DataFrame instead of the precision value
    
    """
    
    hypotheses_types = ["simulation", "abstraction"]
    hypotheses = hypotheses_types if hypothesis == "both" else [hypothesis]
    
    stat_dict = {}
    
    if include_dubtrials == "only":
        hypothesis = "simulation"
        other_hypothesis = [h for h in hypotheses_types if h != hypothesis][0]

        target_trials = df[
            (df['ball_change'] == True) & # Only trials with a target
            (df['accuracy'] != None) & # Only trials with an accuracy value
            (df[hypothesis[:3] + '_congruent'] == True) & # Only trials congruent with the hypothesis
            (df[other_hypothesis[:3] + '_congruent'] == True) # Only trials congruent with the other hypothesis
        ]
        output = np.mean(target_trials['accuracy']) if not return_df else target_trials
        stat_dict["sim + abs"] = output
    else:
    
        for hypothesis in hypotheses:
            other_hypothesis = [h for h in hypotheses_types if h != hypothesis][0]
            

            target_trials = df[
                (df['ball_change'] == True) & # Only trials with a target
                (df['accuracy'] != None) & # Only trials with an accuracy value
                (df[hypothesis[:3] + '_congruent'] == True) & # Only trials congruent with the hypothesis
                (~df[other_hypothesis[:3] + '_congruent']) # Only trials incongruent with the other hypothesis
            ]
            if include_dubtrials:
                target_trials = df[
                    (df['ball_change'] == True) & # Only trials with a target
                    (df['accuracy'] != None) & # Only trials with an accuracy value
                    (df[hypothesis[:3] + '_congruent'] == True)] # Only trials congruent with the hypothesis
                
            output = np.mean(target_trials['accuracy']) if not return_df else target_trials
            stat_dict[hypothesis] = output
        
    return stat_dict
    
    
def get_accuracy(df, hypothesis:str = "both", include_dubtrials:bool | str=False, return_df:bool=False):
    """Accuracy: (True positives + True negatives) / Total cases
    Args:
        df (pd.dataframe): The data
        hypothesis (str): The hypothesis to test. Can be "simulation", "abstraction" or "both"
        include_dubtrials (bool): Whether to include trials where both hypotheses are congruent
        return_df (bool): Whether to return the filtered DataFrame instead of the accuracy value
    """
    
    hypotheses_types = ["simulation", "abstraction"]
    hypotheses = hypotheses_types if hypothesis == "both" else [hypothesis]
    
    stat_dict = {}
    
    if include_dubtrials == "only":
        hypothesis = "simulation"
        other_hypothesis = [h for h in hypotheses_types if h != hypothesis][0]

        relevant_trials = df[
            (df['accuracy'] != None) & # Only trials with an accuracy value
            (df[hypothesis[:3] + '_congruent'] == True) & # Only trials congruent with the hypothesis
            (df[other_hypothesis[:3] + '_congruent'] == True) # Only trials congruent with the other hypothesis
        ]
        
        true_positives = relevant_trials[(relevant_trials['ball_change'] == True) & (relevant_trials['accuracy'] == 1)].shape[0]
        true_negatives = relevant_trials[(relevant_trials['ball_change'] == False) & (relevant_trials['accuracy'].isnull())].shape[0]
        total_cases = relevant_trials.shape[0]
        
        accuracy = (true_positives + true_negatives) / total_cases if total_cases > 0 else 0
        output = accuracy if not return_df else relevant_trials
        stat_dict["sim + abs"] = output
    else:
        for hypothesis in hypotheses:
            other_hypothesis = [h for h in hypotheses_types if h != hypothesis][0]
            
            relevant_trials = df[
                (df['accuracy'] != None) & # Only trials with an accuracy value
                (df[hypothesis[:3] + '_congruent'] == True) & # Only trials congruent with the hypothesis
                (~df[other_hypothesis[:3] + '_congruent']) # Only trials incongruent with the other hypothesis
            ]
            if include_dubtrials:
                relevant_trials = df[
                    (df['accuracy'] != None) & # Only trials with an accuracy value
                    (df[hypothesis[:3] + '_congruent'] == True)] # Only trials congruent with the hypothesis
            
            true_positives = relevant_trials[(relevant_trials['ball_change'] == True) & (relevant_trials['accuracy'] == 1)].shape[0]
            true_negatives = relevant_trials[(relevant_trials['ball_change'] == False) & (relevant_trials['accuracy'].isnull())].shape[0]
            total_cases = relevant_trials.shape[0]
            
            accuracy = (true_positives + true_negatives) / total_cases if total_cases > 0 else 0
            output = accuracy if not return_df else relevant_trials
            stat_dict[hypothesis] = output
    
    return stat_dict

    
    
def get_f1_score(df, hypothesis:str = "both", include_dubtrials=False, return_df:bool=False):
    """F1 score: 2 * (precision * recall) / (precision + recall)
    Args:
        df (pd.dataframe): The data
        hypothesis (str): The hypothesis to test. Can be either "simulation", "abstraction" or "both"
        include_dubtrials (bool): Whether to include trials where both hypotheses are congruent
        return_df (bool): Whether to return the filtered DataFrame instead of the precision value
    
    """
    
    precision = get_precision(df, hypothesis, include_dubtrials, return_df)
    recall = get_sensitivity(df, hypothesis, include_dubtrials, return_df)
    
    f1 = {k: 2 * (precision[k] * recall[k]) / (precision[k] + recall[k]) for k in precision.keys()}
    
    return f1

def filter_condition(df:pd.DataFrame, sim_con:bool | None, expol_con:bool | None):
    """Filter the DataFrame based on the simulation and motion extrapolation congruence

    Args:
        df (pd.DataFrame): Original dataframe
        sim_con (bool): Congruency with simulation hypothesis
        expol_con (bool): Congruency with motion extrapolation hypothesis

    Returns:
        pd.DataFrame: The filtered dataframe
    """    
    sim_filt_df = df if sim_con is None else df[df["sim_congruent"] == sim_con]
    expol_filt_df = sim_filt_df if expol_con is None else sim_filt_df[sim_filt_df["abs_congruent"] == expol_con]
    
    return expol_filt_df


def get_rt(df, 
           sim_con:bool,
           expol_con:bool,
           return_df:bool=False, 
           only_correct:bool=False):
    """Precision: True positives / (True positives + False negatives)
    Args:
        df (pd.dataframe): The data
        hypothesis (str): The hypothesis to test. Can be either "simulation", "abstraction" or "both"
        include_dubtrials (bool): Whether to include trials where both hypotheses are congruent
        return_df (bool): Whether to return the filtered DataFrame instead of the precision value
    
    """
        
    cond_filt_df = filter_condition(df, sim_con, expol_con)

    
    df_filtered = cond_filt_df[(cond_filt_df['accuracy'].notnull()) & # If a response was given
                     (cond_filt_df['ball_change'] == True)] # If a target was shown

    output = np.mean(df_filtered['rt']) if not return_df else df_filtered
        
    return output

def get_precision(df, sim_con, expol_con, return_df):
    """Precision: True positives / (True positives + False negatives)
    Args:
        df (pd.DataFrame): The data
        sim_con (bool): Simulation condition
        expol_con (bool): Expol condition
        return_df (bool): Whether to return the filtered DataFrame
    """
    cond_filt_df = filter_condition(df, sim_con, expol_con)
    
    # Ensure the indices match
    cond_filt_df = cond_filt_df.reindex(df.index)
    
    df_filtered = df[(cond_filt_df['accuracy'].notnull()) &  # If a response was given
                     (cond_filt_df['response'].notnull())]  # If a target was shown

    output = np.mean(df_filtered['accuracy']) if not return_df else df_filtered

    return output

def get_hit_rate(df, 
           sim_con:bool,
           expol_con:bool,
           return_df:bool=False, 
           only_correct:bool=False):
    """Precision: True positives / (True positives + False negatives)
    Args:
        df (pd.dataframe): The data
        hypothesis (str): The hypothesis to test. Can be either "simulation", "abstraction" or "both"
        include_dubtrials (bool): Whether to include trials where both hypotheses are congruent
        return_df (bool): Whether to return the filtered DataFrame instead of the precision value
    
    """
    
    cond_filt_df = filter_condition(df, sim_con, expol_con)
            
    targets_df = cond_filt_df[cond_filt_df['ball_change'] == True] # If a target was shown

    output = np.mean(targets_df['accuracy']) if not return_df else targets_df
        
    return output


def cleanup_accuracy(df):
    """Sets the accuracy to NaN if the ball color change is approximately 0 or if the response time is NaN.

    Args:
        df (pd.DataFrame): The filthy df

    Returns:
        clean_df (pd.DataFrame): The clean df
    """   
    clean_df = df.copy()  # Create a copy of the input DataFrame to avoid modifying it directly
     
    # Ensure the "accuracy" column is of a compatible dtype (e.g., float)
    if df["accuracy"].dtype == bool:
        df["accuracy"] = df["accuracy"].astype(float)
    
    # Set "accuracy" to NaN where "ball_color_change" is approximately 0
    df.loc[(df["ball_color_change"] > -0.0001) & (df["ball_color_change"] < 0.0001), "accuracy"] = np.nan
    df.loc[df["rt"].isna(), "accuracy"] = np.nan # Fine because it's a 2AFC so row should be excluded.

    return clean_df
  
def cleanup_response(df):
    # Set "response" to None where "rt" is NaN
    df.loc[df["rt"].isna(), "response"] = None
    return df
  
def post_hoc_rt(df):
    """Compute rt based on the absolute rt and the target onset.

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        df (pd.DataFrame): df with the computed rt as novel column
    """    
    
    df["comp_rt"] = df["absolute_rt"] - df["target_onset"]
    # Turn negative values to NaN (in case response absent)
    df.loc[df["comp_rt"] < 0, "comp_rt"] = np.nan
    return df

def filter_by_target_onset(df, threshold):
    """
    Filters rows in the DataFrame where 'target_onset' differs from the mean of the first 20 values
    by more than the specified threshold.

    Args:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): The maximum allowed difference from the mean.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    # Compute the mean of the first 20 values in the 'target_onset' column
    mean_target_onset = df["target_onset"].iloc[:20].mean()

    # Filter rows where the absolute difference exceeds the threshold
    filtered_df = df[abs(df["target_onset"] - mean_target_onset) <= threshold]

    return filtered_df

def flip_response_and_accuracy(df):
    """
    Flips the values in the 'response' column ('brighter' <-> 'darker') 
    and inverts the 'accuracy' column (True <-> False).

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame with flipped 'response' and 'accuracy'.
    """
    # Flip 'response' column
    response_mapping = {"brighter": "darker", "darker": "brighter"}
    df["response"] = df["response"].map(response_mapping).fillna(df["response"])

    # Flip 'accuracy' column
    df["accuracy"] = df["accuracy"].apply(lambda x: not x if pd.notna(x) else x)

    return df

def filter_by_slope(raw_df, slope_threshold, exclude_before=0, exclude_after=0, consecutive_threshold=1, verbose=False):
    """
    Filters out rows from a DataFrame based on the slope of the 'target_onset' column.
    Sets 'accuracy', 'response', 'rt', and 'target_onset' to None for rows where the slope exceeds the threshold
    for a specified number of consecutive trials.

    Args:
        raw_df (pd.DataFrame): The input DataFrame.
        slope_threshold (float): The slope threshold to detect steep increases.
        exclude_before (int): Number of trials to exclude before the detected trial.
        exclude_after (int): Number of trials to exclude after the detected trial.
        consecutive_threshold (int): Number of consecutive trials where the slope must exceed the threshold.
        verbose (bool): If True, prints the number of trials excluded.

    Returns:
        pd.DataFrame: The modified DataFrame with filtered rows.
    """
    df = raw_df.copy()  # Create a copy of the input DataFrame to avoid modifying it directly
    
    # Calculate the slope (difference between consecutive rows in 'target_onset')
    df["slope"] = df["target_onset"].diff()

    # Identify indices where the slope exceeds the threshold
    steep_slope_indices = df.index[df["slope"] > slope_threshold].tolist()

    # Group consecutive indices and filter only those with sufficient length
    consecutive_groups = []
    current_group = []

    for idx in steep_slope_indices:
        if not current_group or idx == current_group[-1] + 1:
            current_group.append(idx)
        else:
            if len(current_group) >= consecutive_threshold:
                consecutive_groups.append(current_group)
            current_group = [idx]

    # Add the last group if it meets the threshold
    if len(current_group) >= consecutive_threshold:
        consecutive_groups.append(current_group)

    # Create a set of indices to filter out
    indices_to_filter = set()
    for group in consecutive_groups:
        for idx in group:
            start_idx = max(0, idx - exclude_before)
            end_idx = min(len(df) - 1, idx + exclude_after)
            indices_to_filter.update(range(start_idx, end_idx + 1))

    # Convert the set to a list before using it as an indexer
    indices_to_filter = list(indices_to_filter)

    # Set 'accuracy', 'response', 'rt', and 'target_onset' to None for the filtered rows
    # df.loc[indices_to_filter, ["accuracy", "response", "rt", "target_onset"]] = None

    # Iterate over the columns to adaptively assign None or np.nan
    for col in ["accuracy", "response", "rt", "target_onset"]:
        # Check if the column's dtype is incompatible with np.nan
        if df[col].dtype == 'bool':
            # Cast the column to float to allow np.nan
            df[col] = df[col].astype(float)
        
        # Assign None or np.nan based on the column's dtype
        if df[col].dtype == 'object':  # If the column is of type object (e.g., strings)
            df.loc[indices_to_filter, col] = None
        else:  # For numeric types (e.g., float, int)
            df.loc[indices_to_filter, col] = np.nan

    # Drop the temporary 'slope' column
    df.drop(columns=["slope"], inplace=True)

    # Print verbose information if enabled
    if verbose:
        print(f"Number of trials excluded: {len(indices_to_filter)}")
        if verbose:
            print(f"Consecutive groups excluded: {consecutive_groups}")

    return df


def adaptive_zscore(raw_df, column="rt", change_threshold=0.1, verbose=False):
    """
    Adaptively z-scores the values in the specified column based on segments of trials
    with stable mean and standard deviation.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to z-score (default is "rt").
        change_threshold (float): The threshold for detecting changes in mean or std.
        verbose (bool): If True, prints information about detected segments.

    Returns:
        pd.Series: A new column with adaptively z-scored values.
    """
    df = raw_df.copy()  # Create a copy of the input DataFrame to avoid modifying it directly
    # Initialize variables
    zscored_values = np.zeros(len(df))
    start_idx = 0

    while start_idx < len(df):
        # Define a sliding window to detect stable segments
        end_idx = start_idx + 1
        while end_idx < len(df):
            # Calculate mean and std for the current segment
            segment = df.iloc[start_idx:end_idx + 1][column]
            mean = segment.mean()
            std = segment.std()

            # Calculate mean and std for the next trial
            next_trial = df.iloc[end_idx + 1:end_idx + 2][column]
            if next_trial.empty:
                break
            next_mean = next_trial.mean()
            next_std = next_trial.std()

            # Check if the change exceeds the threshold
            if abs(next_mean - mean) > change_threshold or abs(next_std - std) > change_threshold:
                break

            end_idx += 1

        # Z-score the current segment
        segment = df.iloc[start_idx:end_idx + 1][column]
        segment_mean = segment.mean()
        segment_std = segment.std()
        zscored_values[start_idx:end_idx + 1] = (segment - segment_mean) / segment_std

        if verbose:
            print(f"Segment {start_idx}-{end_idx}: mean={segment_mean:.2f}, std={segment_std:.2f}")

        # Move to the next segment
        start_idx = end_idx + 1

    return pd.Series(zscored_values, index=df.index)

# Define the mapping function
def transform_ball_color_change(df):
    df['rank'] = df['ball_color_change'].rank(method='dense')

    mapping = {
        1: -2,
        2: -1,
        3: 0,
        4: 1,
        5: 2
    }
    df['ball_color_change'] = df['rank'].map(mapping)
    return df


############ A LOT OF THIS COULD BE PUT INTO THE PLOTTING.PY FILE, BUT FOR NOW IT'S OKAY #####

from functions.plotting import curve_fit, logistic_function, polyfit, gaussian, polyval
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def filter_data_for_hypothesis(data, split_hypotheses, cmap="gist_earth", n_bins=7, custom_par_split:bool=False):
    """
    Filter data based on the hypothesis type requested.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing trial data
    split_hypotheses : str
        Type of hypothesis split to apply
    cmap : str
        Matplotlib colormap name to use for color assignments
        
    Returns:
    --------
    list of tuples
        Each tuple contains (dataset, label, color) for a hypothesis
    """
    # This assumes filter_condition is defined elsewhere in your codebase
    from functions.plotting import filter_condition  # Import locally to avoid circular imports
    
    # Extract data for each condition
    cc = filter_condition(data, True, True)
    ci = filter_condition(data, True, False)
    ic = filter_condition(data, False, True)
    ii = filter_condition(data, False, False)
    
    # Get colormap from the argument
    color_spectrum = plt.get_cmap(cmap)
    
    # Define all possible hypotheses
    all_hypotheses = ['CC', 'CI', 'IC', 'II'] if split_hypotheses in ["all", "bounce_surprise", "continue_surprise", "interactor_trials", "empty_trials", "end_position"] else ['sim_exp', 'sim_unexp']
    if split_hypotheses == "start_color" or custom_par_split:
        n_bins = 7 if n_bins is None else n_bins
        all_hypotheses = [f'Lum {i+1}' for i in range(n_bins + 1)]
    # Normalize indices for colormap
    norm = mcolors.Normalize(vmin=0, vmax=len(all_hypotheses) - 1)
    
    # Return appropriate datasets based on split_hypotheses
    if split_hypotheses == "all":
        return [
            (cc, 'SC_EC', color_spectrum(norm(0))),
            (ci, 'SC_EI', color_spectrum(norm(1))),
            (ic, 'SI_EC', color_spectrum(norm(2))),
            (ii, 'SI_EI', color_spectrum(norm(3)))
        ]
    elif split_hypotheses == "sim_surprise":
        sim_exp = pd.concat([cc, ci])
        sim_unexp = pd.concat([ic, ii])
        return [
            (sim_exp, 'Sim con', color_spectrum(norm(0))),
            (sim_unexp, 'Sim incon', color_spectrum(norm(1))),
        ]
    elif split_hypotheses == "expol_surprise":
        expol_exp = pd.concat([cc, ic])
        expol_unexp = pd.concat([ci, ii])
        return [
            (expol_exp, 'Expol con', color_spectrum(norm(0))),
            (expol_unexp, 'Expol incon', color_spectrum(norm(1))),
        ]
    elif split_hypotheses == "bounce_surprise":
        return [
            (ci, 'Bounce exp', color_spectrum(norm(1))),
            (ii, 'Bounce unexp', color_spectrum(norm(3))),
        ]
    elif split_hypotheses == "continue_surprise":
        return [
            (cc, 'Continue exp', color_spectrum(norm(0))),
            (ic, 'Continue unexp', color_spectrum(norm(2))),
        ]
    elif split_hypotheses == "interactor_presence":
        int_present = data[data["trial_type"] == "interactor"]
        int_absent = data[data["trial_type"] == "empty"]
        return [
            (int_present, 'Interactor present', color_spectrum(norm(0))),
            (int_absent, 'Interactor absent', color_spectrum(norm(1))),
        ]
    elif split_hypotheses == "interactor_trials":
        return [
            (ci, 'Bounce off line', color_spectrum(norm(1))),
            (ic, 'Pass through line', color_spectrum(norm(2))),
        ]
    elif split_hypotheses == "empty_trials":
        return [
            (cc, 'Pass through nothing', color_spectrum(norm(0))),
            (ii, 'Bounce off nothing', color_spectrum(norm(3))),
        ]
    elif split_hypotheses == "end_position":
        # Assuming 'end_position' is a column in your DataFrame
        return [
            (data[data['end_pos'] == 'left'], 'Left', color_spectrum(norm(0))),
            (data[data['end_pos'] == 'right'], 'Right', color_spectrum(norm(1))),
            (data[data['end_pos'] == 'up'], 'Up', color_spectrum(norm(2))),
            (data[data['end_pos'] == 'down'], 'Down', color_spectrum(norm(3))),
        ]
    
    elif split_hypotheses == "start_color":
        # Create quantile-based bins for the start color
        quantiles = np.linspace(0, 1, n_bins + 1)  # Define quantiles
        bins = data['ball_start_color'].quantile(quantiles)  # Get bin edges based on quantiles
        # Create labels for the bins
        bin_labels = [f'Lum {i+1}' for i in range(n_bins)]
        # Bin the data
        data['start_color_bin'] = pd.cut(data['ball_start_color'], bins=bins, labels=bin_labels, include_lowest=True)
        # Now filter the data based on these bins
        binned_data = []
        for i in range(n_bins):
            bin_data = data[data['start_color_bin'] == bin_labels[i]]
            binned_data.append((bin_data, bin_labels[i], color_spectrum(norm(i))))
        return binned_data
    elif custom_par_split:
        # Create quantile-based bins for the start color
        quantiles = np.linspace(0, 1, n_bins + 1)  # Define quantiles
        bins = data[split_hypotheses].quantile(quantiles)  # Get bin edges based on quantiles
        # Create labels for the bins
        bin_labels = [f'Custom{i+1}' for i in range(n_bins)]
        # Bin the data
        data['start_color_bin'] = pd.cut(data[split_hypotheses], bins=bins, labels=bin_labels, include_lowest=True)
        # Now filter the data based on these bins
        binned_data = []
        for i in range(n_bins):
            bin_data = data[data['start_color_bin'] == bin_labels[i]]
            binned_data.append((bin_data, bin_labels[i], color_spectrum(norm(i))))
        return binned_data
    
    else:
        return []


def calculate_response_probability(subset):
    """
    Calculate the probability of "brighter" responses for a data subset.
    
    Parameters:
    -----------
    subset : pandas.DataFrame
        DataFrame containing trial responses
        
    Returns:
    --------
    float
        Probability of "brighter" response
    """
    if len(subset) > 0:
        n_valid_responses = subset['response'].isin(['brighter', 'darker'])
        n_brighter = (subset.loc[n_valid_responses, 'response'] == 'brighter').sum()
        return n_brighter / n_valid_responses.sum() if n_valid_responses.sum() > 0 else np.nan
    return np.nan


def extract_data_metrics(data_subset, color_changes, rt_stat="rt"):
    """
    Extract probability and reaction time metrics for a subset of data.
    
    Parameters:
    -----------
    data_subset : pandas.DataFrame
        DataFrame containing trial data
    color_changes : list
        List of color change values to analyze
    rt_stat : str
        Which reaction time stat to use
        
    Returns:
    --------
    tuple
        (color_changes, probabilities, mean_rts, rt_errors)
    """
    valid_changes = []
    prob_brighter = []
    mean_rt = []
    rt_error = []
    
    for change in color_changes:
        subset = data_subset[data_subset['ball_color_change'] == change]
        if len(subset) > 0:
            valid_changes.append(change)
            prob_brighter.append(calculate_response_probability(subset))
            mean_rt.append(subset[rt_stat].mean())
            rt_error.append(subset[rt_stat].std() / np.sqrt(len(subset)))
    
    return valid_changes, prob_brighter, mean_rt, rt_error


def fit_sigmoid_curve(x_values, y_values):
    """
    Fit a sigmoid (logistic) function to data points.
    
    Parameters:
    -----------
    x_values : array-like
        X values for fitting
    y_values : array-like
        Y values for fitting
        
    Returns:
    --------
    tuple
        (x_fit, y_fit, parameters) or (None, None, None) if fitting fails
    """
    try:
        # Filter out NaN values
        valid_indices = ~np.isnan(y_values)
        valid_x = x_values[valid_indices]
        valid_y = y_values[valid_indices]
        
        if len(valid_x) < 4:  # Need at least 4 points for reliable sigmoid
            return None, None, None
        
        # Initial parameter guesses: L=1, x0=midpoint, k=1, b=0
        p0 = [1, np.median(valid_x), 1, 0]
        
        # Parameter bounds
        bounds = ([0.9, min(valid_x), 0.001, -0.1], 
                 [1.1, max(valid_x), 10, 0.1])
        
        # Fit the sigmoid function
        params, _ = curve_fit(logistic_function, valid_x, valid_y, 
                             p0=p0, bounds=bounds, maxfev=10000)
        
        # Create smooth curve with fine-grained x values
        x_fit = np.linspace(min(valid_x), max(valid_x), 100)
        y_fit = logistic_function(x_fit, *params)
        
        return x_fit, y_fit, params
    except Exception as e:
        print(f"Error fitting sigmoid: {e}")
        return None, None, None


def fit_polynomial_curve(x_values, y_values, degree):
    """
    Fit a polynomial of specified degree to data points.
    
    Parameters:
    -----------
    x_values : array-like
        X values for fitting
    y_values : array-like
        Y values for fitting
    degree : int
        Degree of polynomial to fit
        
    Returns:
    --------
    tuple
        (x_fit, y_fit) or (None, None) if fitting fails
    """
    try:
        # Filter out NaN values
        valid_indices = ~np.isnan(y_values)
        valid_x = x_values[valid_indices]
        valid_y = y_values[valid_indices]
        
        if len(valid_x) < degree + 1:  # Need at least degree + 1 points
            return None, None
        
        # Fit polynomial
        coefs = polyfit(valid_x, valid_y, degree)
        
        # Create smooth curve with fine-grained x values
        x_fit = np.linspace(min(valid_x), max(valid_x), 100)
        y_fit = polyval(x_fit, coefs)
        
        return x_fit, y_fit
    except Exception as e:
        print(f"Error fitting polynomial: {e}")
        return None, None


def fit_gaussian_curve(x_values, y_values):
    """
    Fit a Gaussian function to data points.
    
    Parameters:
    -----------
    x_values : array-like
        X values for fitting
    y_values : array-like
        Y values for fitting
        
    Returns:
    --------
    tuple
        (x_fit, y_fit) or (None, None) if fitting fails
    """
    try:
        # Filter out NaN values
        valid_indices = ~np.isnan(y_values)
        valid_x = x_values[valid_indices]
        valid_y = y_values[valid_indices]
        
        if len(valid_x) < 3:  # Need at least 3 points
            return None, None
        
        # Initial parameter guesses (amplitude, mean, std)
        p0 = [np.max(valid_y), np.mean(valid_x), np.std(valid_x)]
        
        # Fit Gaussian function
        popt, _ = curve_fit(gaussian, valid_x, valid_y, p0=p0)
        
        # Create smooth curve with fine-grained x values
        x_fit = np.linspace(min(valid_x), max(valid_x), 100)
        y_fit = gaussian(x_fit, *popt)
        
        return x_fit, y_fit
    except Exception as e:
        print(f"Error fitting Gaussian: {e}")
        return None, None


def setup_figure(figdims=(16, 8)):
    """
    Create and set up the figure and axes for plotting.
    
    Parameters:
    -----------
    figdims : tuple
        Figure dimensions (width, height)
        
    Returns:
    --------
    tuple
        (figure, axes1, axes2)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figdims)
    plt.tight_layout(pad=4)
    
    # Configure probability plot (ax1)
    ax1.set_xlabel('Ball Luminance Change')
    ax1.set_ylabel('Probability of "brighter" Response')
    ax1.set_title('Psychometric Function: Probability of "brighter" Response')
    ax1.set_ylim(0, 1)
    ax1.yaxis.grid(True, alpha=.25)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xticks([-2, -1, 0, 1, 2])
    ax1.set_xticklabels(['darkest', 'darker', 'same', 'brighter', 'brightest'])
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlim(-2, 2)
    
    # Configure reaction time plot (ax2)
    ax2.set_xlabel('Ball Luminance Change')
    ax2.set_ylabel('Reaction Time (s)')
    ax2.set_title('Reaction Time by Ball Lumin Change')
    ax2.yaxis.grid(True, alpha=.25)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlim(-2, 2)
    ax2.set_xticks([-2, -1, 0, 1, 2])
    ax2.set_xticklabels(['darkest', 'darker', 'same', 'brighter', 'brightest'])
    
    return fig, ax1, ax2


def plot_single_dataset(data, axes, prob_poly_fit=None, rt_poly_fit=None, rt_stat="rt"):
    """
    Plot data for a single dataset (no hypothesis splitting).
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing trial data
    axes : tuple
        (ax1, ax2) matplotlib axes for plotting
    prob_poly_fit : int or None
        Use sigmoid fit for probability data if not None
    rt_poly_fit : int or None
        Degree of polynomial for RT data if not None
    rt_stat : str
        Which reaction time stat to use
    """
    ax1, ax2 = axes
    
    # Extract unique color changes and convert to numeric
    color_changes = sorted(data['ball_color_change'].unique())
    x_numeric = np.array([float(x) for x in color_changes])
    
    # Extract metrics
    _, prob_brighter, mean_rt, rt_error = extract_data_metrics(data, color_changes, rt_stat)
    
    # Plot probability data
    ax1.plot(color_changes, prob_brighter, 'o-', color='blue', markersize=8, label='Data')
    
    # Add sigmoid fit if requested
    if prob_poly_fit is not None and len(color_changes) > 3:
        x_fit, y_fit, params = fit_sigmoid_curve(x_numeric, np.array(prob_brighter))
        if x_fit is not None:
            L, x0, k, b = params
            ax1.plot(x_fit, y_fit, 'r--', linewidth=2, 
                    label=f'Sigmoid fit\nMidpoint: {x0:.2f}, Slope: {k:.2f}')
    
    # Plot reaction time data
    ax2.errorbar(color_changes, mean_rt, yerr=rt_error, fmt='o-', color='green', 
                markersize=8, capsize=5, label='Data')
    
    # Add polynomial fit if requested
    if rt_poly_fit is not None and len(color_changes) > rt_poly_fit:
        x_fit, y_fit = fit_polynomial_curve(x_numeric, np.array(mean_rt), rt_poly_fit)
        if x_fit is not None:
            ax2.plot(x_fit, y_fit, 'r--', linewidth=2, 
                    label=f'{rt_poly_fit}-degree polynomial fit')
    
    # Add trial count annotations
    for i, change in enumerate(color_changes):
        count = len(data[data['ball_color_change'] == change])
        ax1.annotate(f'n={count}', xy=(change, prob_brighter[i]), 
                    xytext=(0, 10), textcoords='offset points', 
                    ha='center', va='bottom', fontsize=8)

def plot_hypothesis_datasets(hypotheses, axes, prob_poly_fit=None, rt_poly_fit=None, 
                            rt_gaussian_fit=None, rt_stat="rt", prob_type="o-", 
                            rt_type="o-", fit_type="--", line_alpha=0.4, 
                            fit_alpha=0.9, line_width=3.5, fit_width=8.5,
                            error_bar_style="standard"):
    """
    Plot data for multiple hypothesis datasets.
    
    Parameters:
    -----------
    hypotheses : list
        List of (dataset, label, color) tuples for each hypothesis
    axes : tuple
        (ax1, ax2) matplotlib axes for plotting
    prob_poly_fit : bool
        Whether to fit sigmoid to probability data
    rt_poly_fit : int or None
        Degree of polynomial to fit to RT data
    rt_gaussian_fit : bool
        Whether to fit Gaussian to RT data
    rt_stat : str
        Which reaction time stat to use
    prob_type, rt_type, fit_type : str
        Line styles for plots
    line_alpha, fit_alpha : float
        Alpha transparency for lines
    line_width, fit_width : float
        Line widths
    error_bar_style : str
        Style of error bars - "standard", "transparent_outline", or "continuous_band"
    """
    ax1, ax2 = axes
    
    # Get all color changes across all hypotheses
    all_changes = set()
    for hyp_data, _, _ in hypotheses:
        all_changes.update(hyp_data['ball_color_change'].unique())
    color_changes = sorted(all_changes)
    
    # Plot each hypothesis dataset
    for hyp_data, label, color in hypotheses:
        # Skip if no data for this hypothesis
        if len(hyp_data) == 0:
            continue
        
        hyp_color_changes = sorted(hyp_data['ball_color_change'].unique())
        if len(hyp_color_changes) == 0:
            continue
            
        # Convert to numeric for fitting
        hyp_x_numeric = np.array([float(x) for x in hyp_color_changes])
        
        # Extract metrics
        _, prob_brighter, mean_rt, rt_error = extract_data_metrics(
            hyp_data, hyp_color_changes, rt_stat)
        
        # Convert lists to numpy arrays
        prob_array = np.array(prob_brighter)
        rt_array = np.array(mean_rt)
        rt_error_array = np.array(rt_error)
        
        # Plot probability data
        ax1.plot(hyp_color_changes, prob_brighter, prob_type, color=color, 
                markersize=6, label=f'{label}', alpha=line_alpha, linewidth=line_width)
        
        # Add sigmoid fit if requested
        if prob_poly_fit is not None:
            x_fit, y_fit, params = fit_sigmoid_curve(hyp_x_numeric, prob_array)
            if x_fit is not None:
                L, x0, k, b = params
                ax1.plot(x_fit, y_fit, fit_type, color=color, linewidth=fit_width, 
                        alpha=fit_alpha, label=f'Slope: {k:.2f}\nMidpoint: {x0:.2f}')
        
        # Plot reaction time data with appropriate error visualization
        if error_bar_style == "continuous_band":
            # First, create a smooth interpolation for the mean line and confidence bands
            # We'll use a cubic spline interpolation for smoothing
            from scipy.interpolate import make_interp_spline, BSpline
            
            # Need at least 4 points for cubic spline
            if len(hyp_x_numeric) >= 4:
                # Create a smoother x-axis with more points
                x_smooth = np.linspace(hyp_x_numeric.min(), hyp_x_numeric.max(), 100)
                
                # Create the spline model for the mean
                try:
                    spl = make_interp_spline(hyp_x_numeric, rt_array, k=3, bc_type="natural")  # cubic spline
                    rt_smooth = spl(x_smooth)
                    
                    # Create splines for upper and lower confidence bounds
                    upper_bound = rt_array + rt_error_array
                    lower_bound = rt_array - rt_error_array
                    
                    # spl_upper = make_interp_spline(hyp_x_numeric, upper_bound, k=3, bc_type=([(3, 0.0)], [(3, 0.0)]))
                    # spl_lower = make_interp_spline(hyp_x_numeric, lower_bound, k=3, bc_type=([(3, 0.0)], [(3, 0.0)]))
                    spl_upper = make_interp_spline(hyp_x_numeric, upper_bound, k=3, bc_type="natural")
                    spl_lower = make_interp_spline(hyp_x_numeric, lower_bound, k=3, bc_type="natural")
                    
                    rt_upper_smooth = spl_upper(x_smooth)
                    rt_lower_smooth = spl_lower(x_smooth)
                    
                    # Plot the confidence band
                    ax2.fill_between(x_smooth, rt_lower_smooth, rt_upper_smooth, 
                                    color=color, alpha=0.25)
                    
                    # Plot the smoothed mean line
                    ax2.plot(x_smooth, rt_smooth, '-', color=color, 
                            alpha=line_alpha, linewidth=line_width, label=f'{label}')
                    
                    # Add markers at the actual data points
                    ax2.plot(hyp_x_numeric, rt_array, 'o', color=color, 
                            markersize=8, alpha=line_alpha)
                    
                except Exception as e:
                    # Fallback to standard plotting if spline fails
                    print(f"Spline interpolation failed: {e}. Falling back to standard plot.")
                    ax2.errorbar(hyp_color_changes, mean_rt, yerr=rt_error, fmt=rt_type, 
                                color=color, markersize=8, capsize=5, 
                                label=f'{label}', alpha=line_alpha, linewidth=line_width)
            else:
                # Not enough points for spline, use standard error bars
                ax2.errorbar(hyp_color_changes, mean_rt, yerr=rt_error, fmt=rt_type, 
                            color=color, markersize=8, capsize=5, 
                            label=f'{label}', alpha=line_alpha, linewidth=line_width)
                
        elif error_bar_style == "transparent_outline":
            # Plot line connecting points
            ax2.plot(hyp_color_changes, mean_rt, rt_type.replace('o', ''), 
                    color=color, alpha=line_alpha, linewidth=line_width, label=f'{label}')
            
            # Plot markers
            ax2.plot(hyp_color_changes, mean_rt, 'o', 
                    color=color, markersize=8, alpha=line_alpha)
            
            # Add transparent error bars as filled regions
            for i, (x, y, err) in enumerate(zip(hyp_color_changes, mean_rt, rt_error)):
                # Create polygon for error region
                y_low = y - err
                y_high = y + err
                
                # Draw error bars with thicker transparent outline
                ax2.fill_between([x-0.1, x+0.1], [y_low, y_low], [y_high, y_high], 
                                color=color, alpha=0.2)
                
                # Add vertical line for error bar
                ax2.plot([x, x], [y_low, y_high], '-', color=color, alpha=0.5, linewidth=2)
        else:
            # Standard error bars
            ax2.errorbar(hyp_color_changes, mean_rt, yerr=rt_error, fmt=rt_type, 
                        color=color, markersize=8, capsize=5, capthick=2, elinewidth=1.5,
                        markerfacecolor='white', markeredgewidth=2,
                        label=f'{label}', alpha=line_alpha, linewidth=line_width)
        
        # Add Gaussian fit if requested
        if rt_gaussian_fit:
            x_fit, y_fit = fit_gaussian_curve(hyp_x_numeric, rt_array)
            if x_fit is not None:
                ax2.plot(x_fit, y_fit, fit_type, color=color, linewidth=fit_width,
                        alpha=fit_alpha, label=f'Gaussian fit')
        
        # Add polynomial fit if requested
        if rt_poly_fit:
            x_fit, y_fit = fit_polynomial_curve(hyp_x_numeric, rt_array, rt_poly_fit)
            if x_fit is not None:
                ax2.plot(x_fit, y_fit, fit_type, color=color, linewidth=fit_width,
                        alpha=fit_alpha, label=f'{rt_poly_fit}deg polynom fit')

def print_hypothesis_raw_values(hypotheses, rt_stat="rt"):
    """
    Print raw values for each hypothesis dataset.
    
    Parameters:
    -----------
    hypotheses : list
        List of (dataset, label, color) tuples for each hypothesis
    rt_stat : str
        Which reaction time stat to use
    """
    for hyp_data, label, color in hypotheses:
        hyp_color_changes = sorted(hyp_data['ball_color_change'].unique())
        
        # Skip if no data for this hypothesis
        if len(hyp_color_changes) == 0:
            continue
        
        # Extract metrics
        _, prob_brighter, mean_rt, rt_error = extract_data_metrics(
            hyp_data, hyp_color_changes, rt_stat)
        
        # Print values
        print(f"Raw values for {label}:")
        print(f"Color changes: {hyp_color_changes}")
        print(f"Probabilities: {prob_brighter}")
        print(f"Reaction times: {mean_rt}")
        print(f"RT errors: {rt_error}")


def create_psychometric_plots(data, title=None, prob_ylim=None, rt_ylim=None, 
                              prob_poly_fit=None, rt_poly_fit=None, rt_gaussian_fit=None, split_hypotheses:str=None,
                              save_fig:bool=False, cmap:str="gist_earth", line_alpha:float=.4, fit_alpha:float=.9,
                              prob_type:str="o-", rt_type:str="o-", fit_type:str="--",
                              rt_correct_only:bool=False, fit_width:float=8.5, line_width:float=3.5,
                              string_x_labels:bool=False, rt_stat:str="rt", figdims:tuple=(16, 8),
                              error_bar_style:str="standard", show_legend:bool=True, n_bins:int=7,
                              custom_par_split:bool=False):
    """
    Create two psychometric plots with customizable features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the columns 'ball_color_change', 'response', and "rt"
    title : str, optional
        Custom title for the overall figure
    prob_ylim : tuple, optional
        Custom y-axis range for probability plot (min, max)
    rt_ylim : tuple, optional
        Custom y-axis range for reaction time plot (min, max)
    prob_poly_fit : int or None, optional
        Use sigmoid fit for probability data (if not None)
    rt_poly_fit : int or None, optional
        Degree of polynomial to fit to reaction time data (None for no fit)
    split_hypotheses : str, optional
        Whether to split the data into four hypotheses using filter_condition, 
        can be "all", "sim_surprise", or "expol_surprise"
    rt_stat : str, optional
        Whether to take normal rt or absolute rt
    error_bar_style : str, optional
        Style for error bars, "standard" or "transparent_outline"
    """
    # Ensure polynomial degrees are integers
    if rt_poly_fit is not None:
        rt_poly_fit = int(rt_poly_fit)
    
    # Set up figure and axes
    fig, ax1, ax2 = setup_figure(figdims)
    
    # Set custom y-axis limits if provided
    if prob_ylim:
        ax1.set_ylim(prob_ylim)
    
    if rt_ylim:
        ax2.set_ylim(rt_ylim)
    
    # Set title
    if title:
        plt.suptitle(title, fontsize=16)
    else:
        plt.suptitle('Psychometric Analysis of Ball Hue Task', fontsize=16)
    
    # Plot data based on whether we're splitting by hypothesis
    if split_hypotheses is None:
        # Plot single dataset
        plot_single_dataset(data, (ax1, ax2), prob_poly_fit, rt_poly_fit, rt_stat)
    else:
        # Get hypothesis datasets and plot them
        hypotheses = filter_data_for_hypothesis(data, split_hypotheses, cmap, n_bins, custom_par_split)
        plot_hypothesis_datasets(
            hypotheses, (ax1, ax2), 
            prob_poly_fit, rt_poly_fit, rt_gaussian_fit, rt_stat,
            prob_type, rt_type, fit_type, line_alpha, fit_alpha,
            line_width, fit_width, error_bar_style
        )
        # Print raw values
        # print_hypothesis_raw_values(hypotheses, rt_stat) # NU EVEN NIET
    
    if show_legend:
        ax1.legend(loc='best')
        ax2.legend(loc='best')
    
    # Save figure if requested
    if save_fig:
        plt.savefig('psychometric_plots.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig