import os
import sys
import numpy as np
import random
from psychopy import visual, core, event
from omegaconf import OmegaConf
import exptools2
from exptools2.core import Session, Trial
from exptools2.core import PylinkEyetrackerSession
import pandas as pd
import os.path as op



def design_spatloc():
    """
    Design the spatial location of the stimuli.
    """
    locations = ["left", "right", "top", "bottom"]
    movement = ["in", "out", "stutter"]
    # Create a list of all combinations of locations and movements
    combinations = [(loc, mov) for loc in locations for mov in movement]
    # Shuffle the combinations
    random.shuffle(combinations)
    # Create a DataFrame with the combinations
    df = pd.DataFrame(combinations, columns=["location", "movement"])
    # Add a column for the trial number
    df["trial_number"] = range(1, len(df) + 1)
    # Add a column for the trial type
    df["trial_type"] = ["spatloc"] * len(df)

    return df
