from mdp_formulation import GazeFormulationBaseClass, low_gaze_config_with_L_M_V
import pdb
import random
import json 
import os
import pickle
from multiprocessing import Process, Queue
import time
import csv
from rewards import low_gaze_reward

def calculate_q_value(q_table, current_state, current_action, reward, next_state_key, config):
    # Calculate the Q-value
    # for state1_key, state1 in q_table.items():
    #     next_state = config.states[state1_key]
    #     # Get the reward for this transition
    #     # Calculate the Q-value
    #     q_value = reward + config.discount_factor * max(q_table[state1_key].values())
    #     return q_value
    max_future_q = max(q_table[next_state_key].values()) if next_state_key in q_table else 0
    q_current = q_table[current_state][current_action]
    
    # Q-learning update rule
    q_new = q_current + config.learning_rate * (reward + config.gamma * max_future_q - q_current)
    q_table[current_state][current_action] = q_new  
    
    return q_new

def get_all_next_states(current_state_key, actions, states):
    current_gaze = states[current_state_key]
    next_states = {}

    for action_key, action_vector in actions.items():
        # Compute the next gaze score based on the action
        next_gaze = current_gaze + sum(action_vector)

        # Ensure gaze score stays within valid range [0, 10]
        next_gaze = max(0, min(10, next_gaze))

        # Find the state key corresponding to the next gaze score
        next_state_key = [k for k, v in states.items() if v == next_gaze][0]  # Pick the first match

        next_states[action_key] = next_state_key

    return next_states
    
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
    print(f"Q-table saved to {filename}")

def train_q_learning(q_table, states, actions, reward_function, episodes=1000, epsilon=0.1):

    state_visits = {s: 0 for s in states.keys()}  # Track how often each state is visited

    for episode in range(episodes):
        # Start from a random state
        current_state = random.choice(list(states.keys()))

        for _ in range(len(states)):  # Iterate through all states at least once per episode
            # Select action using epsilon-greedy strategy
            if random.uniform(0, 1) < epsilon:
                current_action = random.choice(list(actions.keys()))  # Explore
            else:
                current_action = max(q_table[current_state], key=q_table[current_state].get)  # Exploit

            # Get possible next states
            next_states = get_all_next_states(current_state, actions, states)
            next_state_key = next_states[current_action]  # Pick the next state for the current action

            # Compute reward
            reward = reward_function(states[current_state], actions[current_action])

            # Update Q-value
            calculate_q_value(q_table, current_state, current_action, reward, next_state_key, low_gaze_config_with_L_M_V)

            # Track visits
            state_visits[current_state] += 1

            # Move to next state
            current_state = next_state_key

    print("State Visit Counts:", state_visits)

# A function to print the highest q_table value for each state
def print_highest_q_values(q_table):
    for state_key, state in q_table.items():
        highest_q_value = max(state.values())
        #get key of highest q value
        highest_q_value_key = max(state, key=state.get)
        print(f"State: {state_key}, Highest Q-value Key: {highest_q_value_key}, Highest Q-value: {highest_q_value}")
            
if __name__=="__main__":
    ## Training routine
    # Build the Q-Table
    ## TODO:: Generalise to arguments
    config = low_gaze_config_with_L_M_V
    # pdb.set_trace()
    q_table = {}
    for state_key in config.states.keys():
        q_table[state_key] = {}
        for action_key in config.actions.keys():
            q_table[state_key][action_key] = 0.0  # Initialize Q-values to 0.0
    
    train_q_learning(q_table, config.states, config.actions, low_gaze_reward, config.episodes, config.epsilon)

    print("Q-table after training:")
    print_highest_q_values(q_table)
    
    # Save the Q-table to a CSV file
    save_q_table_to_csv(q_table, 'q_table_check.csv')
    
    # # run through the q-table
    # for episode in range(config.episodes):
    #     print(f"Episode: {episode}")
    #     for state_key, state in q_table.items():
    #         for action_key, action in state.items():
    #             current_state = config.states[state_key]
    #             current_action = config.actions[action_key]
    #             reward = config.reward_function(current_state, current_action)
    #             # Calculate the Q-value
    #             q_value = calculate_q_value(q_table, current_state, current_action, reward, config)
    #             q_table[state_key][action_key] = q_value
            

                    

            