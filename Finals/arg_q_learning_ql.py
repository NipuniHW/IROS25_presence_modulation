import argparse
from mdp_formulation import GazeFormulationBaseClass, low_gaze_config_with_L_M_V, low_gaze_config, medium_gaze_config, high_gaze_config
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
        max_future_q = max(q_table[state1_key].values())
        # q_current = q_table.get(str(current_state), {}).get(current_action, 0.0)
        q_current = q_table[current_state][current_action]
        # pdb.set_trace() 
        # Q-learning update rule
        q_new = q_current + config.learning_rate * (reward + config.gamma * max_future_q - q_current)
        q_table[current_state][current_action] = q_new  
        
        # q_value = reward + config.discount_factor * max(q_table[state1_key].values())
        return q_new
    
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
    parser = argparse.ArgumentParser(description='Q-Learning Configuration')
    parser.add_argument('--config', type=str, choices=['low_gaze_config_with_L_M_V', 'low_gaze_config', 'medium_gaze_config', 'high_gaze_config'], required=True, help='Choose the configuration')
    parser.add_argument('--csv', type=str, required=True, help='CSV file name to save the Q-table')
    args = parser.parse_args()

    # Choose the configuration based on the argument
    if args.config == 'low_gaze_config_with_L_M_V':
        config = low_gaze_config_with_L_M_V
    elif args.config == 'low_gaze_config':
        config = low_gaze_config
    elif args.config == 'medium_gaze_config':
        config = medium_gaze_config
    elif args.config == 'high_gaze_config':
        config = high_gaze_config

    ## Training routine
    # Build the Q-Table
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
                q_value = calculate_q_value(q_table, state_key, action_key, reward, config)
                q_table[state_key][action_key] = q_value
    
    # Save the Q-table to a CSV file
    save_q_table_to_csv(q_table, args.csv)