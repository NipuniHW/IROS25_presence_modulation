from mdp_formulation import GazeFormulationBaseClass, medium_gaze_config
import pdb
import random
import json 
import os
import pickle
from multiprocessing import Process, Queue
import time
import csv

def calculate_q_value(q_table, current_state, current_action, reward, config):
    # Calculate the Q-value
    for state1_key, state1 in q_table.items():
        next_state = config.states[state1_key]
        # Get the reward for this transition
        # Calculate the Q-value
        q_value = reward + config.discount_factor * max(q_table[state1_key].values())
        return q_value
    
def save_q_table_to_csv(q_table, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        header = ['State'] + list(next(iter(q_table.values())).keys())
        writer.writerow(header)
        # Write the Q-table rows
        for state, actions in q_table.items():
            row = [state] + list(actions.values())
            writer.writerow(row)

if __name__=="__main__":
    ## Training routine
    # Build the Q-Table
    ## TODO:: Generalise to arguments
    config = medium_gaze_config
    q_table = {}
    for state_key in config.states.keys():
        q_table[state_key] = {}
        for action_key in config.actions.keys():
            q_table[state_key][action_key] = 0.0  # Initialize Q-values to 0.0

    # run through the q-table
    for episode in range(config.episodes):
        print(f"Episode: {episode}")
        for state_key, state in q_table.items():
            for action_key, action in state.items():
                current_state = config.states[state_key]
                current_action = config.actions[action_key]
                reward = config.reward_function(current_state, current_action)
                # Calculate the Q-value
                q_value = calculate_q_value(q_table, current_state, current_action, reward, config)
                q_table[state_key][action_key] = q_value
    
    # Save the Q-table to a CSV file
    save_q_table_to_csv(q_table, 'q_table_medium.csv')