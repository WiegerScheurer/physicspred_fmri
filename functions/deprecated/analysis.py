import os
import sys
import numpy as np
import pandas as pd

# class BallData:
#     def __init__(self):
#         self.data = None
#         self.datadir = None
#         self.subject = None
#         self.task = None
#         self.subs = None

# TODO: TURN INTO EFFICIENT CLASS, AT SOME POINT, BUT NOW TOO EARLY

def get_data(subject:str | None=None, datadir: str | None=None, task: str | None = "ball_hiccup"):
    datadir = "/Users/wiegerscheurer/repos/physicspred/data" if datadir is None else datadir
    
    subs = [sub for sub in os.listdir(datadir) if sub.startswith("sub")] if subject is None else [subject]
        

    file_stack = []
    for sub in subs:

        datafiles = os.listdir(f"{datadir}/{sub}/{task}/")
        for file in datafiles:
        
            if file.endswith(".csv") and not file.startswith("design_matrix"):
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
