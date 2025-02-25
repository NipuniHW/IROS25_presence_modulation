import csv
import numpy as np
import pandas as pd
import yaml
import random
from mdp_formulation import high_gaze_config_6

def load_q_table(file_path):
    q_table = pd.read_csv(file_path)
    q_table.index = q_table.index.astype(int)
    q_table.set_index(q_table.columns[0], inplace=True)
    q_table.index.name = "State"
    return q_table

# Function to choose an action based on the current state
def choose_action(state, q_table):
    c_state = str(state)
    best_action = max(q_table[c_state], key=q_table[c_state].get) 

    # chosen_action = np.random.choice(best_actions)  # Randomly select among best actions if tie

    # Ensure the chosen action is split into a list of three components
    return best_action  # Split by ", " to separate the values

def save_trajectory_ep_to_yaml(testing_run_name, subject_count, training_dict):
    save_path = f'{testing_run_name}/{testing_run_name}_subject_{subject_count}_trajectory.yaml'
    with open(save_path, 'w') as file:
        yaml.dump(training_dict, file)
    return

def save_trajectory_ep_to_yaml_2(testing_run_name, training_dict):
    save_path = f'{testing_run_name}/{testing_run_name}_first_1_min.yaml'
    with open(save_path, 'w') as file:
        yaml.dump(training_dict, file)
    return

def save_trajectory_ep_to_yaml_3(testing_runname, training_dict):
    save_path = f'{testing_runname}/{testing_runname}_dynamic.yaml'
    with open(save_path, 'w') as file:
        yaml.dump(training_dict, file)
    return

def load_training_state(training_run_name):
    with open(f'{training_run_name}/{training_run_name}_training_state.yaml', 'r') as file:
        training_state = yaml.load(file, Loader=yaml.FullLoader)
    episode = training_state['episode']
    epsilon = training_state['epsilon']
    q_table_name = training_state['q_table_name']
    q_table = load_q_table(q_table_name)
    return q_table, episode, epsilon

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

def load_q_table_from_csv(filename):
    q_table = {}
    with open(filename
    , mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            state = row[0]
            actions = {header[i]: float(row[i]) for i in range(1, len(row))}
            q_table[state] = actions
    return q_table

            
if __name__=='__main__':
    config = high_gaze_config_6
    q_table_save_name = 'saved_q_table'
    q_table_init = {}
    for state_key in config.states.keys():
        q_table_init[state_key] = {}
        for action_key in config.actions.keys():
            # set the q table to a random value between 0 and 1
            q_table_init[state_key][action_key] = random.random()
    save_q_table_to_csv(q_table_init, q_table_save_name)
    q_table_loaded = load_q_table_from_csv(q_table_save_name)
    if q_table_loaded == q_table_init:
        print('Q-table saved and loaded successfully')
    else:
        print('Error in the saving and loading of the Q-table')
    