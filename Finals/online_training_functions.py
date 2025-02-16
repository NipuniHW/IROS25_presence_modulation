import csv
import yaml
from mdp_formulation import GazeFormulationBaseClass, low_gaze_config_with_L_M_V, low_gaze_config, medium_gaze_config, high_gaze_config
import random
import json

def save_training_state_after_episode(q_table, episode, training_run_name):
    q_table_name = f'{training_run_name}/{training_run_name}_episode_{episode}.csv'
    # write the q_table_name and episode count to a yaml file
    training_state = {
        'q_table_name': q_table_name,
        'episode': episode
    }
    with open(f'{training_run_name}/{training_run_name}_training_state.yaml', 'w') as file:
        yaml.dump(training_state, file)
    # save the q_table to a csv file
    save_q_table_to_csv(q_table, q_table_name)
    return

def load_training_state(training_run_name):
    with open(f'{training_run_name}/{training_run_name}_training_state.yaml', 'r') as file:
        training_state = yaml.load(file, Loader=yaml.FullLoader)
    episode = training_state['episode']
    q_table_name = training_state['q_table_name']
    q_table = load_q_table_from_csv(q_table_name)
    return q_table, episode

def create_empty_q_table(config):
    q_table = {}
    for state_key in config.states.keys():
        q_table[state_key] = {}
        for action_key in config.actions.keys():
            q_table[state_key][action_key] = 0.0
    return q_table

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
    config = high_gaze_config
    q_table_save_name = 'q_table_test.csv'
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
        print('I think there is an error in the saving and loading of the Q-table')
    

