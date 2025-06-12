from datetime import datetime
import random
import os
from scipy.stats import truncexpon
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import pandas as pd
import colour
import pandas as pd
import numpy as np
import random
from itertools import product

import yaml
from types import SimpleNamespace

def load_config(yaml_file):
    """Load YAML configuration file and return as nested SimpleNamespace objects."""
    with open(yaml_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    return dict_to_namespace(config_dict)

def dict_to_namespace(d):
    """Recursively convert dictionary to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d

def bellshape_sample(mean, sd, n_samples, plot:bool=False, shuffle:bool=True):
    
    sample_pool = np.array([random.normalvariate(mean, sd) for sample in range(n_samples)])
    
    if shuffle:
        random.shuffle(sample_pool)
    else:
        sample_pool.sort()
    if plot:
        plt.hist(sample_pool, bins=50)
    
    return list(sample_pool)

# # Use for the different target hues, balance over it.
# def ordinal_sample(mean, step_size, n_elements, plot:bool=False, round_decimals:int | None=None, 
#                    pos_bias_factor:float=1.0, neg_bias_factor:float=1.0):
#     # Calculate the start and end points
#     half_range = (n_elements - 1) // 2
#     start = mean - half_range * step_size
#     end = mean + half_range * step_size
    
#     # Generate the steps
#     steps = np.arange(start, end + step_size, step_size)
    
#     # Ensure the correct number of elements
#     if len(steps) > n_elements:
#         steps = steps[:n_elements]
        
#     # Round the steps to 2 decimal places
#     if round_decimals is not None:
#         steps = np.round(steps, round_decimals)
        
#     # Apply positive and negative bias factors
#     if pos_bias_factor != 1.0:
#         steps = [step * pos_bias_factor if step > mean else step for step in steps]
#     if neg_bias_factor != 1.0:
#         steps = [step * neg_bias_factor if step < mean else step for step in steps]
    
#     return steps


# Use for the different target hues, balance over it.
def ordinal_sample(mean, step_size, n_elements, round_decimals:int | None=None, 
                   pos_bias_factor:float=1.0, neg_bias_factor:float=1.0, 
                   pos_bias_shift:float=0.0, neg_bias_shift:float=0.0):
    
    # Calculate the start and end points
    half_range = (n_elements - 1) // 2
    start = mean - half_range * step_size
    end = mean + half_range * step_size
    
    # Generate the steps
    steps = np.arange(start, end + step_size, step_size)
    
    # Ensure the correct number of elements
    if len(steps) > n_elements:
        steps = steps[:n_elements]
        

        
    # Apply positive and negative bias factors
    if pos_bias_factor != 1.0 or pos_bias_shift != 0.0:
        steps = [step * pos_bias_factor + pos_bias_shift if step > mean else step for step in steps]
    if neg_bias_factor != 1.0 or neg_bias_shift != 0.0:
        # steps = [step * neg_bias_factor if step < mean else step for step in steps]
        steps = [step * neg_bias_factor - neg_bias_shift if step < mean else step for step in steps]
    
        # Round the steps to 2 decimal places
    if round_decimals is not None:
        steps = np.round(steps, round_decimals)
    
    return steps

# Compound function to be used in psychopy (make sure this is also usable in other projects, as quite important)
def oklab_to_rgb(oklab, psychopy_rgb:bool=False):
    # Convert OKLab to XYZ
    xyz = colour.Oklab_to_XYZ(oklab)
    # Convert XYZ to RGB
    rgb = [np.clip(((rgb_idx * 2) - 1), -1, 1) for rgb_idx in colour.XYZ_to_sRGB(xyz)] if psychopy_rgb else colour.XYZ_to_sRGB(xyz)

    return rgb

def michelson_contrast(l_max:float, l_min:float):
    """Function to compute the Michelson contrast between two luminances.

    Args:
        lum1 (float): Luminance of object 1
        lum2 (float): Luminance of object 2

    Returns:
        float: Michelson contrast between the two objects
    """
    return (l_max - l_min) / (l_max + l_min)

def equal_contrasts(darker_object:float, startlum_light_obj:float, light_increase:float, delta_lum:bool=False):
    """Function to compute how much darker a light object on a dark background needs to become to
    match the difference in contrast with background of the original object becoming brighter. 

    Assuming that the ball cannot become darker than the background, the function computes the luminance of the
    If it does, return an error (but implement still)
    Args:
        darker_object (float): The luminance of the darker object, such as the background
        startlum_light_obj (float): The start luminance of the brighter object
        light_increase (float): The increase in luminance of the brighter object

    Returns:
        float: The luminance of the darker object that would match the contrast difference of the brighter object
    """

    start_contrast = michelson_contrast(startlum_light_obj, darker_object)
    
    brighter_obj = startlum_light_obj + light_increase # Think about whether this is sensible (not adaptive valence?)
    
    brighter_contrast = michelson_contrast(brighter_obj, darker_object)
    
    brighter_contrast_diff = brighter_contrast - start_contrast
    
    darker_contrast = start_contrast - brighter_contrast_diff
    
    darker_ball_lum = darker_object * ((1 + darker_contrast) / (1 - darker_contrast))
    
    if darker_ball_lum < darker_object:
        print("Error: Darker object is darker than background")
    else:
        return (darker_ball_lum - startlum_light_obj) if delta_lum else darker_ball_lum



############## NEW ONES THAT WORK AND BALANCE NICELY, DON'T USE THE NONE ARGUMENT FOR THE FIRST FUNCTION


def create_balanced_trial_design(trial_n=None, change_ratio:list = [True, False], 
                                 ball_color_change_mean=0, ball_color_change_sd=0.05, startball_lum=.75, background_lum=.25,
                                 neg_bias_factor:float=1.5, neg_bias_shift:float=0.0):
    
    def _clean_trial_options(df):
        # For each row, if trial_option starts with "none", keep only the first 6 characters
        df['trial_option'] = df['trial_option'].apply(
            lambda x: x[:6] if x.startswith('none_') else x
        )
        return df

    # Your options
    interactor_trial_options = ["45_top_r", "45_top_u", "45_bottom_l", "45_bottom_d",
                               "135_top_l", "135_top_u", "135_bottom_r", "135_bottom_d"]
    # empty_trial_options = ["none_l", "none_r", "none_u", "none_d"] * 2
    # Option 1: Create 8 truly unique empty trial options
    empty_trial_options = ["none_l_1", "none_r_1", "none_u_1", "none_d_1", 
                        "none_l_2", "none_r_2", "none_u_2", "none_d_2"]


    directions = ["left", "right"] * 4
    random.shuffle(directions)
    
    # Update the direction mapping
    direction_mapping = {
        "none_l_1": directions[0], "none_l_2": directions[0],
        "none_r_1": directions[1], "none_r_2": directions[1],
        "none_u_1": directions[2], "none_u_2": directions[2],
        "none_d_1": directions[3], "none_d_2": directions[3]
    }
    
    bounce_options = [True, False]
    # ball_change_options = [True, False]
    ball_change_options = change_ratio
    
    # Strangely enough it appears that darker balls should be less extreme than brighter balls. 
    
    ball_color_change_options = list(ordinal_sample(ball_color_change_mean, ball_color_change_sd, n_elements=5, round_decimals=3,
                                     pos_bias_factor=1.0, neg_bias_factor=neg_bias_factor, neg_bias_shift=neg_bias_shift))

    # If trial_n is specified, create a balanced subset
    if trial_n is not None:
        # Make sure trial_n is even for interactor:empty balance
        if trial_n % 2 == 1:
            trial_n -= 1
            print(f"Adjusted trial count to {trial_n} to maintain balance")
        
        half_n = trial_n // 2  # Half for interactor, half for empty
        
        # Create dataframe to store the balanced design
        all_trials = []
        
        # For interactor trials
        # First, create all possible combinations
        interactor_combos = list(product(
            interactor_trial_options,
            bounce_options,
            ball_change_options,
            ball_color_change_options
        ))
        random.shuffle(interactor_combos)  # Shuffle to avoid bias
        
        # Now intelligently select a subset that maximizes balance
        selected_interactor = []
        option_counts = {option: 0 for option in interactor_trial_options}
        bounce_counts = {True: 0, False: 0}
        change_counts = {True: 0, False: 0}
        luminance_counts = {luminance: 0 for luminance in ball_color_change_options}
        
        # First pass: try to get at least one of each option
        for option in interactor_trial_options:
            matching_combos = [c for c in interactor_combos if c[0] == option and c not in selected_interactor]
            if matching_combos:
                selected_interactor.append(matching_combos[0])
                option_counts[option] += 1
                bounce_counts[matching_combos[0][1]] += 1
                change_counts[matching_combos[0][2]] += 1
                luminance_counts[matching_combos[0][3]] += 1
        
        # Second pass: fill in remaining slots balancing bounce and ball_change
        remaining_slots = half_n - len(selected_interactor)
        while remaining_slots > 0:
            # Prioritize by least common option, then bounce, then ball_change
            min_option_count = min(option_counts.values())
            min_options = [opt for opt, count in option_counts.items() if count == min_option_count]
            
            min_bounce_count = min(bounce_counts.values())
            min_bounce = [b for b, count in bounce_counts.items() if count == min_bounce_count]
            
            min_change_count = min(change_counts.values())
            min_change = [c for c, count in change_counts.items() if count == min_change_count]
            
            # Find combos that match our criteria
            matching_combos = [c for c in interactor_combos 
                              if c[0] in min_options 
                              and c[1] in min_bounce 
                              and c[2] in min_change 
                              and c not in selected_interactor]
            
            # If no perfect match, relax constraints one by one
            if not matching_combos:
                matching_combos = [c for c in interactor_combos 
                                  if c[0] in min_options 
                                  and c[1] in min_bounce 
                                  and c not in selected_interactor]
            
            if not matching_combos:
                matching_combos = [c for c in interactor_combos 
                                  if c[0] in min_options 
                                  and c not in selected_interactor]
            
            if not matching_combos:
                matching_combos = [c for c in interactor_combos if c not in selected_interactor]
            
            if matching_combos:
                best_combo = matching_combos[0]
                selected_interactor.append(best_combo)
                option_counts[best_combo[0]] += 1
                bounce_counts[best_combo[1]] += 1
                change_counts[best_combo[2]] += 1
                luminance_counts[best_combo[3]] += 1
                remaining_slots -= 1
            else:
                # If we somehow run out of unique combinations
                break
        
        # Create the interactor trials from our selection
        for trial_option, bounce, ball_change, ball_luminance in selected_interactor:
            all_trials.append({
                'trial_type': 'interactor',
                'trial_option': trial_option,
                'bounce': bounce,
                'phant_bounce_direction': None,
                'ball_change': ball_change,
                'ball_luminance': ball_luminance
            })
        
        # For empty trials, use the same approach
        empty_combos = list(product(
            empty_trial_options,
            bounce_options,
            ball_change_options,
            ball_color_change_options
        ))
        random.shuffle(empty_combos)  # Shuffle to avoid bias
        
        # Now intelligently select a subset that maximizes balance
        selected_empty = []
        option_counts = {option: 0 for option in empty_trial_options}
        bounce_counts = {True: 0, False: 0}
        change_counts = {True: 0, False: 0}
        luminance_counts = {luminance: 0 for luminance in ball_color_change_options}
        
        # First pass: try to get at least one of each option
        for option in empty_trial_options:
            matching_combos = [c for c in empty_combos if c[0] == option and c not in selected_empty]
            if matching_combos:
                selected_empty.append(matching_combos[0])
                option_counts[option] += 1
                bounce_counts[matching_combos[0][1]] += 1
                change_counts[matching_combos[0][2]] += 1
                luminance_counts[matching_combos[0][3]] += 1
        
        # Second pass: fill in remaining slots balancing bounce and ball_change
        remaining_slots = half_n - len(selected_empty)
        while remaining_slots > 0:
            # Prioritize by least common option, then bounce, then ball_change
            min_option_count = min(option_counts.values())
            min_options = [opt for opt, count in option_counts.items() if count == min_option_count]
            
            min_bounce_count = min(bounce_counts.values())
            min_bounce = [b for b, count in bounce_counts.items() if count == min_bounce_count]
            
            min_change_count = min(change_counts.values())
            min_change = [c for c, count in change_counts.items() if count == min_change_count]
            
            # Find combos that match our criteria
            matching_combos = [c for c in empty_combos 
                              if c[0] in min_options 
                              and c[1] in min_bounce 
                              and c[2] in min_change 
                              and c not in selected_empty]
            
            # If no perfect match, relax constraints one by one
            if not matching_combos:
                matching_combos = [c for c in empty_combos 
                                  if c[0] in min_options 
                                  and c[1] in min_bounce 
                                  and c not in selected_empty]
            
            if not matching_combos:
                matching_combos = [c for c in empty_combos 
                                  if c[0] in min_options 
                                  and c not in selected_empty]
            
            if not matching_combos:
                matching_combos = [c for c in empty_combos if c not in selected_empty]
            
            if matching_combos:
                best_combo = matching_combos[0]
                selected_empty.append(best_combo)
                option_counts[best_combo[0]] += 1
                bounce_counts[best_combo[1]] += 1
                change_counts[best_combo[2]] += 1
                luminance_counts[best_combo[3]] += 1
                remaining_slots -= 1
            else:
                # If we somehow run out of unique combinations
                break
        
        # Create the empty trials from our selection
        for trial_option, bounce, ball_change, ball_luminance in selected_empty:
            bounce_direction = direction_mapping[trial_option] if bounce else None
            all_trials.append({
                'trial_type': 'empty',
                'trial_option': trial_option,
                'bounce': bounce,
                'phant_bounce_direction': bounce_direction,
                'ball_change': ball_change,
                'ball_luminance': ball_luminance
            })
        
        # Convert to dataframe and shuffle
        df = pd.DataFrame(all_trials)
        df.sample(frac=1).reset_index(drop=True)
        return _clean_trial_options(df)
    
    # If trial_n is None, create the full balanced design
    else:
        # Create all possible combinations
        all_trials = []
        
        # For interactor trials
        for combo in product(interactor_trial_options, bounce_options, ball_change_options, ball_color_change_options):
            trial_option, bounce, ball_change, ball_luminance = combo
            all_trials.append({
                'trial_type': 'interactor',
                'trial_option': trial_option,
                'bounce': bounce,
                'phant_bounce_direction': None,
                'ball_change': ball_change,
                'ball_luminance': ball_luminance
            })
        
        # For empty trials - we need to duplicate these to match interactor count
        for combo in product(empty_trial_options, bounce_options, ball_change_options, ball_color_change_options):
            trial_option, bounce, ball_change, ball_luminance = combo
            bounce_direction = direction_mapping[trial_option] if bounce else None
            
            # Each empty trial combination needs to appear twice to balance with interactor trials
            # for _ in range(2):
            all_trials.append({
                'trial_type': 'empty',
                'trial_option': trial_option,
                'bounce': bounce,
                'phant_bounce_direction': bounce_direction,
                'ball_change': ball_change,
                'ball_luminance': ball_luminance
            })
    
    
        # Convert to dataframe and shuffle
        df = pd.DataFrame(all_trials)

        df.sample(frac=1).reset_index(drop=True)
        
        return _clean_trial_options(df)
            
        # return output_df.sample(frac=1).reset_index(drop=True)
# TODO: ADAPT SO THAT THE DARKER BALLS ARE NOT SO EXTREME AND THETHE LIGHTER BALLS ARE A TINY BIT MORE EXTREME. 
# ALSO CHECK IF THIS EMPIRICAL FINDING BASED ON 1 SINGLE PILOT PARTICIPANTS ALIGNS WITH WHAT IS KNOWN ABOUT THE
# NONLINEARITY OF THE LUMINANCE FUNCTION.
def build_design_matrix(n_trials:int, change_ratio:list=[True, False], 
                        ball_color_change_mean:float=.45, ball_color_change_sd:float=.05, 
                        trials_per_fullmx:int | None=None, verbose:bool=False,
                        neg_bias_factor:float=1.5, neg_bias_shift:float=0.0):
    """
    Build a design matrix for a given number of trials.

    Parameters:
    - n_trials (int): The total number of trials.
    - verbose (bool): Whether to print verbose output.

    Returns:
    - design_matrix (pd.DataFrame): The resulting design matrix.
    """
    # trials_per_fullmx = 192
    if trials_per_fullmx is None:
        test_dm = create_balanced_trial_design(trial_n=None, 
                                               change_ratio=change_ratio, 
                                               ball_color_change_mean=ball_color_change_mean, 
                                               ball_color_change_sd=ball_color_change_sd,
                                               neg_bias_factor=neg_bias_factor,
                                               neg_bias_shift=neg_bias_shift)
        trials_per_fullmx = len(test_dm)    
        print(f"Number of trials per full matrix: {trials_per_fullmx}")

    full_matrices = n_trials // trials_per_fullmx
    remainder = n_trials % trials_per_fullmx
    
    if verbose:
        print(f"Design matrix for {n_trials} trials, constituting {full_matrices} fully balanced matrices and {remainder} trials balanced approximately optimal.")
    
    if remainder > 0:
        initial_dm = create_balanced_trial_design(remainder, change_ratio=change_ratio, 
                                                  ball_color_change_mean=ball_color_change_mean, 
                                                  ball_color_change_sd=ball_color_change_sd,
                                                  neg_bias_factor=neg_bias_factor,
                                                  neg_bias_shift=neg_bias_shift)
    else:
        initial_dm = pd.DataFrame()
    
    for full_matrix in range(full_matrices + 1):
        dm = create_balanced_trial_design(192, neg_bias_factor=neg_bias_factor)
        dm = create_balanced_trial_design(trials_per_fullmx, change_ratio=change_ratio, 
                                          ball_color_change_mean=ball_color_change_mean, 
                                          ball_color_change_sd=ball_color_change_sd,
                                          neg_bias_factor=neg_bias_factor,
                                          neg_bias_shift=neg_bias_shift)
        if full_matrix == 0:
            design_matrix = initial_dm
        else:
            design_matrix = pd.concat([design_matrix, dm])
            
    # Shuffle the rows and reset the index
    design_matrix = design_matrix.sample(frac=1).reset_index(drop=True)
    return design_matrix

def check_balance(df):
    print(f"Total trials: {len(df)}")
    
    # Check trial type balance
    type_counts = df['trial_type'].value_counts()
    print("\nTrial type balance:")
    print(type_counts)
    
    # Check trial option balance within each trial type
    print("\nTrial option balance for interactor trials:")
    interactor_options = df[df['trial_type'] == 'interactor']['trial_option'].value_counts().sort_index()
    print(interactor_options)
    print(f"Variance: {interactor_options.var():.2f}")
    
    print("\nTrial option balance for empty trials:")
    empty_options = df[df['trial_type'] == 'empty']['trial_option'].value_counts().sort_index()
    print(empty_options)
    print(f"Variance: {empty_options.var():.2f}")
    
    # Check bounce balance
    bounce_counts = df['bounce'].value_counts()
    print("\nBounce balance:")
    print(bounce_counts)
    
    # Check ball change balance
    ball_change_counts = df['ball_change'].value_counts()
    print("\nBall change balance:")
    print(ball_change_counts)
    
    # Check ball luminance balance
    print("\nBall luminance balance:")
    print(df['ball_luminance'].value_counts().sort_index())
    
    print("\nCross-tabulation of bounce × ball_luminance_change:")
    print(pd.crosstab(df['bounce'], df['ball_luminance']))
    
    # Cross-tabulations for more detailed balance checks
    print("\nCross-tabulation of trial_type × bounce:")
    print(pd.crosstab(df['trial_type'], df['bounce']))
    
    print("\nCross-tabulation of trial_type × ball_change:")
    print(pd.crosstab(df['trial_type'], df['ball_change']))
    
    print("\nCross-tabulation of bounce × ball_change:")
    print(pd.crosstab(df['bounce'], df['ball_change']))
    
    # Check balance at the deepest level
    print("\nBalance within interactor trial types:")
    for option in sorted(df[df['trial_type'] == 'interactor']['trial_option'].unique()):
        subset = df[(df['trial_type'] == 'interactor') & (df['trial_option'] == option)]
        print(f"\n{option}:")
        print(f"  Total: {len(subset)}")
        print(f"  Bounce: {subset['bounce'].value_counts().to_dict()}")
        print(f"  Ball Change: {subset['ball_change'].value_counts().to_dict()}")
        print(f"  Ball luminance: {subset['ball_luminance'].value_counts().to_dict()}")
    
    print("\nBalance within empty trial types:")
    for option in sorted(df[df['trial_type'] == 'empty']['trial_option'].unique()):
        subset = df[(df['trial_type'] == 'empty') & (df['trial_option'] == option)]
        print(f"\n{option}:")
        print(f"  Total: {len(subset)}")
        print(f"  Bounce: {subset['bounce'].value_counts().to_dict()}")
        print(f"  Ball Change: {subset['ball_change'].value_counts().to_dict()}")
        print(f"  Ball luminance: {subset['ball_luminance'].value_counts().to_dict()}")

def count_list_types(list):
    """
    Counts the occurrences of each element in a list and returns a dictionary with the counts.

    Args:
        list (list): The input list.

    Returns:
        dict: A dictionary where the keys are the elements in the list and the values are the counts of each element.
    """
    return {i: list.count(i) for i in list}


def setup_folders(subject_id, task_name, base_dir="/Users/wiegerscheurer/repos/physicspred/database"):
    """
    Creates the necessary folders for a given subject and task.

    Args:
        subject_id (str): The ID of the subject.
        task_name (str): The name of the task.

    Returns:
        str: The path to the task directory.
    """

    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    subject_dir = os.path.join(base_dir, subject_id)
    task_dir = os.path.join(subject_dir, task_name)

    # Create directories if they don't exist
    os.makedirs(task_dir, exist_ok=True)

    return task_dir


def save_performance_data(subject_id, task_name, data, design_matrix:bool=False,
                          intermediate:bool=False, base_dir="/Users/wiegerscheurer/repos/physicspred/database",
                          session="1"):
    """
    Save performance data to a CSV file.

    Args:
        subject_id (str): The ID of the subject.
        task_name (str): The name of the task.
        data (pandas.DataFrame): The performance data to be saved.

    Returns:
        None
    """
    task_dir = setup_folders(f"{subject_id}", task_name, base_dir=base_dir)
    #date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # filename = f"{date_str}-ses{session}.csv" if not design_matrix else "design_matrix.csv"
    filename = f"session-{session}.csv" if not design_matrix else "design_matrix.csv"
    filename = f"intermediate.csv" if intermediate else filename
    filepath = os.path.join(task_dir, filename)

    # Save the DataFrame to a CSV file
    data.to_csv(filepath, index=False, float_format="%.8f")


def interpolate_color(start_color, end_color, factor):
    """
    Interpolates between two colors based on a given factor.

    Args:
        start_color (tuple): The starting color as a tuple of RGB values.
        end_color (tuple): The ending color as a tuple of RGB values.
        factor (float): The interpolation factor between 0 and 1.

    Returns:
        tuple: The interpolated color as a tuple of RGB values.
    """
    return start_color + (end_color - start_color) * factor


def determine_sequence(n_trials: int, options: list, randomised: bool = True) -> list:
    """
    Balances a sequence of trials based on the number of trials and options.

    Parameters:
    n_trials (int): The number of trials to balance.
    options (list): The options to balance.
    randomised (bool): Whether to randomise the sequence.

    Returns:
    list: The balanced sequence of trials.

    """

    n_options = len(options)
    n_per_option = n_trials // n_options
    remainder = n_trials % n_options

    balanced_sequence = options * n_per_option + options[:remainder]

    if randomised:
        random.shuffle(balanced_sequence)

    return balanced_sequence


def get_pos_and_dirs(ball_speed, square_size, ball_spawn_spread, ball_speed_change, ball_radius):
    # Possible starting positions
    # start_positions = {
    #     "up": (0, square_size // ball_spawn_spread),
    #     "down": (0, -square_size // ball_spawn_spread),
    #     "left": (-square_size // ball_spawn_spread, 0),
    #     "right": (square_size // ball_spawn_spread, 0),
    # }
    out_of_bounds = (square_size // 2) + (ball_radius)
    
    start_positions = {
        "up": (0, out_of_bounds),
        "down": (0, -out_of_bounds),
        "left": (-out_of_bounds, 0),
        "right": (out_of_bounds, 0),
    }

    # Base directions
    directions = {
        "up": (0, -ball_speed),
        "down": (0, ball_speed),
        "left": (ball_speed, 0),
        "right": (-ball_speed, 0),
    }
    # Fast directions
    fast_ball_speed = ball_speed * ball_speed_change
    fast_directions = {
        "up": (0, -fast_ball_speed),
        "down": (0, fast_ball_speed),
        "left": (fast_ball_speed, 0),
        "right": (-fast_ball_speed, 0),
    }

    # Slow directions
    slow_ball_speed = ball_speed / ball_speed_change
    slow_directions = {
        "up": (0, -slow_ball_speed),
        "down": (0, slow_ball_speed),
        "left": (slow_ball_speed, 0),
        "right": (-slow_ball_speed, 0),
    }
    
    # Skip directions
    skip_ball_speed = ball_speed * (ball_speed_change * 10)
    skip_directions = {
        "up": (0, -skip_ball_speed),
        "down": (0, skip_ball_speed),
        "left": (skip_ball_speed, 0),
        "right": (-skip_ball_speed, 0),
    }
    
    # Wait directions
    wait_ball_speed = ball_speed * (ball_speed_change / 10)
    wait_directions = {
        "up": (0, -wait_ball_speed),
        "down": (0, wait_ball_speed),
        "left": (wait_ball_speed, 0),
        "right": (-wait_ball_speed, 0),
    }
    

    return start_positions, directions, fast_directions, slow_directions, skip_directions, wait_directions


def truncated_exponential_decay(min_iti, truncation_cutoff, size=1000):
    """
    Generate a truncated exponential decay distribution.

    Parameters:
        min_iti (float): The minimum ITI (lower bound of the distribution).
        truncation_cutoff (float): The upper bound of the distribution.
        size (int): Number of samples to generate.

    Returns:
        samples (numpy.ndarray): Random samples from the truncated exponential distribution.
    """
    # Define the scale parameter for the exponential decay
    scale = 1.0  # Adjust this to control the steepness of decay
    b = (truncation_cutoff - min_iti) / scale  # Shape parameter for truncation

    # Generate random samples
    samples = truncexpon(b=b, loc=min_iti, scale=scale).rvs(size=size)
    return samples

def two_sided_truncated_exponential(center, min_jitter, max_jitter, scale=1.0, size=1000):
    """
    Generate a two-sided truncated exponential decay distribution that peaks at the center.

    Parameters:
        center (float): The central point of the distribution (e.g., critical event time).
        min_jitter (float): The minimum jitter value (left bound).
        max_jitter (float): The maximum jitter value (right bound).
        scale (float): The scale parameter controlling steepness of the decay.
        size (int): Number of samples to generate.

    Returns:
        samples (numpy.ndarray): Random samples from the two-sided truncated exponential distribution.
    """
    # Create an array of possible jitter values
    x = np.linspace(min_jitter, max_jitter, 1000)
    
    # Define left and right exponential decays
    left_decay = np.exp(-(center - x[x <= center]) / scale)
    right_decay = np.exp(-(x[x > center] - center) / scale)
    
    # Combine left and right sides
    pdf = np.concatenate([left_decay, right_decay])
    
    # Normalize PDF so it integrates to 1
    pdf /= np.sum(pdf)
    
    # Sample from this custom PDF using inverse transform sampling
    cdf = np.cumsum(pdf)  # Compute cumulative density function
    cdf /= cdf[-1]  # Ensure CDF ends at 1
    random_values = np.random.rand(size)  # Uniform random values between 0 and 1
    samples = np.interp(random_values, cdf, x)  # Map random values to jitter times
    
    return samples

def plot_distribution(samples, min_iti, truncation_cutoff):
    """
    Plot the histogram of the truncated exponential distribution.

    Parameters:
        samples (numpy.ndarray): Random samples from the truncated exponential distribution.
        min_iti (float): The minimum ITI.
        truncation_cutoff (float): The upper bound of the distribution.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=30, density=True, alpha=0.6, color='blue', label="Sampled Data")

    # Plot theoretical PDF for comparison
    scale = 1.0
    b = (truncation_cutoff - min_iti) / scale
    x = np.linspace(min_iti, truncation_cutoff, 100)
    pdf = truncexpon(b=b, loc=min_iti, scale=scale).pdf(x)
    plt.plot(x, pdf, 'r-', lw=2, label="Theoretical PDF")

    plt.title("Truncated Exponential Decay Distribution")
    plt.xlabel("Interval")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_two_sided_distribution(samples, center_time, min_jitter, max_jitter, scale=1.0):
    # Plot histogram of sampled data
    plt.hist(samples, bins=30, density=True, alpha=0.6, color='blue', label="Sampled Data")

    # Plot theoretical PDF for visualization
    x = np.linspace(min_jitter, max_jitter, 1000)
    left_decay = np.exp(-(center_time - x[x <= center_time]) / scale)
    right_decay = np.exp(-(x[x > center_time] - center_time) / scale)
    pdf = np.concatenate([left_decay, right_decay])
    pdf /= np.sum(pdf) * (x[1] - x[0])  # Normalize for plotting purposes
    plt.plot(x, pdf, 'r-', lw=2, label="Theoretical PDF")

    plt.title("Two-Sided Truncated Exponential Decay")
    plt.xlabel("Time relative to center (s)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

def balance_over_bool(boolean_list:list, value_options:list, randomised:bool=True) -> list:
    """Map one list of value options onto the True values of a boolean list.

    Args:
        boolean_list (list): List that indicates which trials should get a value.
        value_options (list): List of the value options
    """    
    
    val_seq = determine_sequence(np.sum(boolean_list), value_options, randomised=randomised)
    
    result = []
    value_index = 0
    for item in boolean_list:
        if item:
            result.append(val_seq[value_index])
            value_index += 1
        else:
            result.append(False)
    return result

def get_phantbounce_sequence(trials:list, rand_bounce_direction_options:list):
    # Make empty nan array of same size as none_l bool
    trial_array = np.empty(len(trials), dtype=object)

    for dir_idx, none_dir in enumerate(["l", "r", "u", "d"]):
        none_dir_bool = [trial == f"none_{none_dir}" for trial in trials]
        trial_array[none_dir_bool] = rand_bounce_direction_options[dir_idx]

    return trial_array
