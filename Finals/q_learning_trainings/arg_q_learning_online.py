import argparse
from copy import deepcopy

import cv2
import yaml
from Finals.mdp_formulation import GazeFormulationBaseClass, low_gaze_config_with_L_M_V, low_gaze_config, medium_gaze_config, high_gaze_config
import pdb
import random
import json 
import os
import pickle
from multiprocessing import Process, Queue
import time
import csv
# from gaze_6s import GazeInterfaceController
from gaze_interface_controller import GazeInterfaceController
from online_training_functions import *
from pepper import *

'''
Online Q-Learning Documentation:
-Ensure you run this code in the Finals folder for convenience

Assumptions: All MDP state transition steps occur at a rate of 3 seconds, if this needs to change we may want to reconsider some implementation details

'''

def calculate_q_value(q_table, previous_state, action, next_state, reward, config):
    
    # Get the reward for this transition
    # Calculate the Q-value
    # convert next state into string
    str_next_state = str(next_state)
    str_prev_state = str(previous_state)
    max_future_q = max(q_table[str_next_state].values())
    # q_current = q_table.get(str(current_state), {}).get(current_action, 0.0)
    q_current = q_table[str_prev_state][action]
    # pdb.set_trace() 
    # Q-learning update rule
    q_new = q_current + config.learning_rate * (reward + config.gamma * max_future_q - q_current)
    q_table[str_prev_state][action] = q_new  
        
    # q_value = reward + config.discount_factor * max(q_table[state1_key].values())
    return q_new
    
def run_training_episode(q_table, config, episode_count, online_episode_duration, epsilon, training_rname):
    print('connecting a session to pepper')
    
    pepper = Pepper()
    # pepper.connect("pepper.local", 9559)
    
    pepper.connect("localhost", 39607)
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
    print('Press Enter to start the training')
    input()
    # start the training
    current_time = time.time()
    
    random_number = random.choice([1, 3, 5, 7, 9])
    light, movement, volume = random_number, random_number, random_number  # Default values 
    #Convert to minutes
    online_episode_duration_seconds = online_episode_duration
    online_episodes_duration_minutes = online_episode_duration*60
    time_step_count = 0
    save_dictionary = {}
    
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
            # Get the current state -- Todo: Convert the gaze score to a state
            previous_state = int(round(gaze_score/10))
            print(f"Current gaze score: {gaze_score} -- giving state:{previous_state}")
            #TODO Choose an action - Integrate
            action, epsilon = choose_action(q_table, previous_state, config, epsilon)
            # TODO:: Send pepper actions here...
            # Update the behavior
            light, movement, volume = pepper.update_behavior(action, light, movement, volume)
            # Get the current gaze score
            gaze_score_ = controller.get_gaze_score()
            next_state = int(round(gaze_score_/10))
            # Get the reward
            reward = config.reward_function(previous_state, action, next_state, config.gaze_threshold)
            # Calculate the Q-value
            q_value = calculate_q_value(q_table, previous_state, action, next_state, reward, config)
            
            save_dictionary['previousstate_episode_' + str(episode_count)+'_timestep_'+str(time_step_count)] = previous_state
            save_dictionary['nextstate_episode_' + str(episode_count)+'_timestep_'+str(time_step_count)] = next_state
            save_dictionary['action_episode_' + str(episode_count)+'_timestep_'+str(time_step_count)] = action
            save_dictionary['reward_episode_' + str(episode_count)+'_timestep_'+str(time_step_count)] = reward
            
            time_step_count+=1
            
            # Update the Q-table
            str_prev_state = str(previous_state)
            q_table[str_prev_state][action] = q_value
            print(f"Next gaze score: {gaze_score_} -- giving state:{next_state}")
        
    save_trajectory_ep_to_yaml(episode_count, training_rname, save_dictionary)
    
    
    print('Training episode complete')
    controller.kill_attention_thread()
    
    del pepper
    return q_table, epsilon

def choose_action(q_table, current_state, config, epsilon):
    # Choose an action
    action_selection_rand = random.uniform(0, 1) 
    c_state = str(current_state)
    if action_selection_rand < epsilon:
        # Explore
        action = random.choice(list(config.actions.keys()))
    else:
        # pdb.set_trace()
        # action = max(q_table[c_state], key=lambda k: q_table[c_state][k])
        # max_value = -1.0
        # action = None
        # for k, v in q_table[c_state].items():
        #     if v > max_value:
        #         max_value = v
        #         action = k
        action = max(q_table[c_state], key=q_table[c_state].get) 
    epsilon*=config.epsilon_decay
    return action, epsilon
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Q-Learning Configuration')
    parser.add_argument('--config', type=str, 
                                    choices=['low_gaze_config_with_L_M_V', 
                                             'low_gaze_config', 
                                             'medium_gaze_config', 
                                             'high_gaze_config'], 
                                             required=True, 
                                             help='Choose the configuration')
    parser.add_argument('--load_training_dir', type=bool, required=True, default=False, help='If you wish to continue training from a saved Q-table, set to true')
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
    epsilon_decay = config.epsilon_decay
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
            q_table, episode_count, epsilon = load_training_state(args.training_runname)
            print('loaded training data from ' + args.training_runname)
        except:
            print('failed to load training data from ' + args.training_runname)
            raise Exception('Failed to load training data from ' + args.training_runname + '. Please check the file path and try again')
    else:
        raise Exception('An invalid arrangment of configurations and training data was provided. Please check the configurations/Arguements and try again')
    
    print('Starting training loop')

    while True:
        # Run an episode

        # Increment the episode count
        episode_count += 1
        # Get the user input asking if they want to continue training
        user_input = input('Would you like to continue training for another episode? (Y/N): ')
        if user_input.lower() == 'y' or user_input.lower() == 'Y':
            # Run the next episode
            q_table, epsilon = run_training_episode(q_table, config, episode_count, online_episode_duration, epsilon,args.training_runname)
            # After episode, save the Q-table to a CSV file
            save_training_state_after_episode(q_table, episode_count, args.training_runname, epsilon)
        else:
            print('Your input was not Y/y. Exiting training')
            break
    
    print('finished mental abuse, yay!!!')