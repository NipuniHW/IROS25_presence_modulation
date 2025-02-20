import csv
import numpy as np
import pandas as pd
import yaml
from mdp_formulation import GazeFormulationBaseClass, low_gaze_config_with_L_M_V, low_gaze_config, medium_gaze_config, high_gaze_config
import random

def load_q_table(file_path):
    q_table = pd.read_csv(file_path)
    q_table.index = q_table.index.astype(int)
    q_table.set_index(q_table.columns[0], inplace=True)
    q_table.index.name = "State"
    return q_table

# Function to choose an action based on the current state
def choose_action(state, q_table):
    if state not in q_table.index:
        raise ValueError(f"State {state} is not in the Q-table")

    action_values = q_table.loc[state]
    max_q_value = action_values.max()
    best_actions = action_values[action_values == max_q_value].index.tolist()

    chosen_action = np.random.choice(best_actions)  # Randomly select among best actions if tie

    # Ensure the chosen action is split into a list of three components
    return chosen_action  # Split by ", " to separate the values
