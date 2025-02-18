import argparse
from copy import deepcopy
import sys
import cv2
import numpy as np
import pandas as pd
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
from online_training_functions import *
from pepper import *

def load_q_table(file_path):
    q_table = pd.read_csv(file_path)
    q_table.index = q_table.index.astype(int)
    q_table.set_index(q_table.columns[0], inplace=True)
    q_table.index.name = "State"
    return q_table

# Function to choose an action based on the current state
def select_action(state, q_table):
    if state not in q_table.index:
        raise ValueError(f"State {state} is not in the Q-table. Available states: {q_table.index.tolist()}")

    action_values = q_table.loc[state]
    max_q_value = action_values.max()
    best_actions = action_values[action_values == max_q_value].index.tolist()

    chosen_action = np.random.choice(best_actions)  # Randomly select among best actions if tie

    # Ensure the chosen action is split into a list of three components
    return chosen_action.split(", ")  # Split by ", " to separate the values

    
def test_q_learning(q_table):
    print('connecting a session to pepper')
    
    pepper = Pepper()
    pepper.connect("pepper.local", 9559)
    
    # pepper.connect("localhost", 41813)
    # Change the camera ID to 2 if using external usb webcam, 0 if using the laptop webcam
    controller = GazeInterfaceController(camera_id=2)
    time.sleep(1)
    # ask the user to press enter to start a calibration
    print('Press Enter to start the calibration')
    input()
    controller.calibration_exe()
    controller.start_detecting_attention()
    # print('Calibration complete')
    # ask the user to press enter to start the training
    time.sleep(1)
    print('Press Enter to start the testing')
    input()
    # start the training
    current_time = time.time()
    
    random_number = random.choice([1, 3, 5, 7, 9])
    light, movement, volume = random_number, random_number, random_number  # Default values 
    #Convert to minutes
    online_episode_duration_seconds = online_episode_duration
    online_episodes_duration_minutes = online_episode_duration*60
    time_step_count = 0
    # save_dictionary = {}
    
    start_time_inner_loop = time.time()
    
    while time.time() - current_time < online_episodes_duration_minutes:
        frame = controller.get_visualisation_frame()
        if frame is not None:
            f = deepcopy(frame)
            # print("the type of frame is ", type(f))
            cv2.imshow('Calibrated HRI Attention Detection', f)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        if time.time() - start_time_inner_loop >= 3:
            start_time_inner_loop = time.time()
            # Get the current gaze score
            gaze_score = controller.get_gaze_score()
            print(f"Current gaze score: {gaze_score}")
            # Get the current state -- Todo: Convert the gaze score to a state
            previous_state = get_gaze_bin(gaze_score)
            # Choose an action - Integrate
            action = select_action(previous_state, q_table)
            # Send pepper actions here...
            # Update the behavior
            light, movement, volume = pepper.update_behavior(action, light, movement, volume)

    
    print('Testing complete')
    controller.kill_attention_thread()
    
    del pepper
    # return q_table, epsilon

    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Q-Learning Configuration')
    parser.add_argument('--config', type=str, 
                                    choices=['low_gaze_config_with_L_M_V', 
                                             'low_gaze_config', 
                                             'medium_gaze_config', 
                                             'high_gaze_config'], 
                                             required=True, 
                                             help='Choose the configuration')
    # parser.add_argument('--load_training_dir', type=bool, required=False, default=False, help='If you wish to continue training from a saved Q-table, set to true')
    # parser.add_argument('--training_runname', type=str, required=True, help='CSV file name to save the Q-table')
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
    epsilon_decay = config.epsilon_decay
    learning_rate = config.learning_rate
    discount_factor = config.discount_factor
    exploration_rate = config.exploration_rate
    episode_count = 0
    online_episode_duration = args.online_episode_duration

    
    if __name__ == "__main__":    
        pepper = Pepper()
        try:
            pepper.connect("pepper.local", 9559)
            if not pepper.is_connected:
                sys.exit(1)
                
            parser = argparse.ArgumentParser(description='Q-Learning Testing')
            parser.add_argument('--q_table', type=str, required=True, help='Path to the Q-table CSV file')
            args = parser.parse_args()

            test_q_learning(args.q_table)
            del pepper 
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Cleaning up...")
            del pepper   
            cv2.destroyAllWindows()
            sys.exit(0)
            
        print('finished mental abuse, yay!!!')