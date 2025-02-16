import argparse

import yaml
from mdp_formulation import GazeFormulationBaseClass, low_gaze_config_with_L_M_V, low_gaze_config, medium_gaze_config, high_gaze_config
import pdb
import random
import json 
import os
import pickle
from multiprocessing import Process, Queue
import time
import csv
from gaze_interface_controller import GazeInterfaceController
from online_training_functions import save_q_table_to_csv, load_q_table_from_csv, create_empty_q_table, save_training_state_after_episode, load_training_state

'''
Online Q-Learning Documentation:
-Ensure you run this code in the Finals folder for convenience

Assumptions: All MDP state transition steps occur at a rate of 3 seconds, if this needs to change we may want to reconsider some implementation details

'''

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
    
def run_training_episode(q_table, config, episode_count, online_episode_duration):
    controller = GazeInterfaceController()
    # ask the user to press enter to start a calibration
    input('Press Enter to start the calibration')
    controller.calibration_exe()
    controller.start_detecting_attention()
    # print('Calibration complete')
    # ask the user to press enter to start the training
    input('Press Enter to start the training')
    # start the training
    current_time = time.time()
    while time.time() - current_time < online_episode_duration:
        # Get the current gaze score
        gaze_score = controller.get_gaze_score()
        # Get the current state
        current_state = config.states_generator(gaze_score)
        # Choose an action
        action = choose_action(q_table, current_state, config)
        # Get the reward
        reward = config.reward_function(current_state, action)
        # Calculate the Q-value
        q_value = calculate_q_value(q_table, current_state, action, reward, config)
        # Update the Q-table
        q_table[current_state][action] = q_value
    
    print('Training episode complete')
    return q_table

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Q-Learning Configuration')
    parser.add_argument('--config', type=str, 
                                    choices=['low_gaze_config_with_L_M_V', 
                                             'low_gaze_config', 
                                             'medium_gaze_config', 
                                             'high_gaze_config'], 
                                             required=True, 
                                             help='Choose the configuration')
    parser.add_argument('--load_training_dir', type=bool, required=False, default=False, help='If you wish to continue training from a saved Q-table, set to true')
    parser.add_argument('--training_runname', type=str, required=True, help='CSV file name to save the Q-table')
    parser.add_argument('--online_episode_duration', type=int, required=True, help='length of a gaze training episode')
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

    # set training information
    epsilon = config.epsilon
    epsilon_decay = config.epislon_decay
    learning_rate = config.learning_rate
    discount_factor = config.discount_factor
    exploration_rate = config.exploration_rate
    episode_count = 0
    online_episode_duration = args.online_episode_duration

    # check if the training folder exists
    # if it does not exist
    if not os.path.exists(args.training_runname) and not args.load_training_dir:
        print('we are executing a new training session and do not need to load training data or configurations')
        os.makedirs(args.training_runname)
        print('made directory:' + args.training_runname + ' for training data')
        # Build the Q-Table
        print('Building Q-table with initial 0 values')
        q_table = create_empty_q_table(config)
        print('New Q-table built for training')
    elif args.load_training_dir and os.path.exists(args.training_runname):
        # load the training data
        # Load the Q-table from the CSV file
        try:
            q_table, episode_count = load_training_state(args.training_runname)
            print('loaded training data from ' + args.training_runname)
        except:
            print('failed to load training data from ' + args.training_runname)
            raise Exception('Failed to load training data from ' + args.training_runname + '. Please check the file path and try again')
    else:
        raise Exception('An invalid arrangment of configurations and training data was provided. Please check the configurations/Arguements and try again')


    print('Starting training loop')

    while True:
        # Run an episode
        

        # After episode, save the Q-table to a CSV file
        save_training_state_after_episode(q_table, episode_count, args.training_runname)

        # Increment the episode count
        episode_count += 1
        # Get the user input asking if they want to continue training
        user_input = input('Would you like to continue training for another episode? (Y/N): ')
        if user_input.lower() == 'Y' or user_input.lower() == 'y':
            continue
        else:
            print('Your input was not Y/y. Exiting training')
            break

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