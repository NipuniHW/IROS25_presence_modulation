import random
from gaze import main
import json 
import os
import json
import pickle
from multiprocessing import Process, Queue
import time
import pdb

# Define the directory where you want to save the files
SAVE_DIR = "/home/nipuni/Documents/Codes/q-learning/Training/low"

# Ensure the directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

gaze_bins = list(range(1, 10)) 
expected_ranges = (0, 30)

actions = [
            ("Increase L", "Increase M", "Increase V"),
            ("Increase L", "Increase M", "Keep V"),
            ("Increase L", "Increase M", "Decrease V"),
            ("Increase L", "Keep M", "Increase V"),
            ("Increase L", "Keep M", "Keep V"),
            ("Increase L", "Keep M", "Decrease V"),
            ("Increase L", "Decrease M", "Increase V"),
            ("Increase L", "Decrease M", "Keep V"),
            ("Increase L", "Decrease M", "Decrease V"),
            ("Keep L", "Increase M", "Increase V"),
            ("Keep L", "Increase M", "Keep V"),
            ("Keep L", "Increase M", "Decrease V"),
            ("Keep L", "Keep M", "Increase V"),
            ("Keep L", "Keep M", "Keep V"),  # No changes
            ("Keep L", "Keep M", "Decrease V"),
            ("Keep L", "Decrease M", "Increase V"),
            ("Keep L", "Decrease M", "Keep V"),
            ("Keep L", "Decrease M", "Decrease V"),
            ("Decrease L", "Increase M", "Increase V"),
            ("Decrease L", "Increase M", "Keep V"),
            ("Decrease L", "Increase M", "Decrease V"),
            ("Decrease L", "Keep M", "Increase V"),
            ("Decrease L", "Keep M", "Keep V"),
            ("Decrease L", "Keep M", "Decrease V"),
            ("Decrease L", "Decrease M", "Increase V"),
            ("Decrease L", "Decrease M", "Keep V"),
            ("Decrease L", "Decrease M", "Decrease V"),
        ]

behavior_levels = list(range(11))  # Levels from 0 to 10

q_table = {}
for gaze_bin in gaze_bins:
            for light in behavior_levels:
                for movement in behavior_levels:
                    for volume in behavior_levels:
                        state = (gaze_bin, light, movement, volume)        
                        q_table[state] = {action: 0 for action in actions} 

pdb.set_trace()
                        
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.9  # Initial exploration rate
epsilon_decay = 0.99
min_epsilon = 0.1
num_episodes = 1000

def get_reward(gaze_score):
    if 0 <= gaze_score <= 30:
        reward = 50
    if 31 <= gaze_score <= 60:
        reward = -10
    if 61 <= gaze_score <= 100:
        reward = -50
    return reward

def select_action(state):
    if not actions:
        raise ValueError("Action list is empty!")

    if state not in q_table or not q_table[state]:  
        return random.choice(actions)  

    q_values = q_table[state]
    
    # Ensure actions have valid Q-values
    if all(value == 0 or value is None for value in q_values.values()):
        return random.choice(actions)  

    return max(q_values, key=q_values.get)  

    
def update_behavior(state, action):
        gaze_bin, light, movement, volume = state
        l_action, m_action, v_action = action   
          
        # print("Debug: Before updating behaviour")  
         
        if l_action == "Increase L":
            light = min(10, light + 1)
        elif l_action == "Decrease L":
            light = max(0, light - 1)        
        if m_action == "Increase M":
            movement = min(10, movement + 1)
        elif m_action == "Decrease M":
            movement = max(0, movement - 1)            
        if v_action == "Increase V":
            volume = min(10, volume + 1)
        elif v_action == "Decrease V":
            volume = max(0, volume - 1)

        # Keep Same
        if l_action == "Keep L":
            light = light
        elif m_action == "Keep M":
            movement = movement
        elif v_action == "Keep V":
            volume = volume
            
        print(f"L, M, V: {light}, {movement}, {volume}")
        return (gaze_bin, light, movement, volume)   
     
def assign_gaze_bin(gaze_score, num_bins=10):
        if not 0 <= gaze_score <= 100:
            raise ValueError("Gaze score must be between 0 and 100.")
        bin_size = 100 / num_bins
        gaze_bin = int(gaze_score // bin_size)
        if gaze_bin == num_bins:
            gaze_bin = num_bins - 1
        return gaze_bin    
    
def q_learning_episode(gaze_score, state):
    global epsilon
    gaze_bin, old_light, old_movement, old_volume = state

    # Initialize new_light, new_movement, and new_volume with current values
    new_light, new_movement, new_volume = old_light, old_movement, old_volume
    action = None
    reward = 0  # Default reward

    try:
        print(f"Current state at episode: {state}\n")

        # Select action
        action = select_action(state)
        print(f"Action selected: {action} \n")

        # Get expected min and max
        expected_min, expected_max = expected_ranges
        print(f"Expected min/ max: {expected_min}, {expected_max} \n")

        # Calculate reward
        reward = get_reward(gaze_score)
        if reward is None:
            raise ValueError("Reward is None")

        # Update behavior and get new state
        _, new_light, new_movement, new_volume = update_behavior(state, action)

        print(f"Old state at episode: {gaze_bin}, {old_light}, {old_movement}, {old_volume} \n")
        print(f"New state at episode: {gaze_bin}, {new_light}, {new_movement}, {new_volume} \n")

    except Exception as e:
        print(f"Error in q_learning_episode: {e}")
        # If an error occurs, return the current state and default reward
        return gaze_bin, new_light, new_movement, new_volume, action, reward

    return gaze_bin, new_light, new_movement, new_volume, action, reward 
          
def train_q_learning():
    global q_table, epsilon
    main_generator = main() 
    
#    time.sleep(12) 
        
    print("Starting Q-learning training...")         
    
    L, M, V = 0, 0, 0 
    previous_state = None
    episode_data = []  
 
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/1000")
        try:
            gaze_score = next(main_generator)        
            print(f"Received gaze score: {gaze_score}")
            
            gaze_bin = assign_gaze_bin(gaze_score)
            episode_data = []
            
            if previous_state is None:
                state = (gaze_bin, L, M, V)
            else:
                state = previous_state
            print(f"State at learning: {state}")
            
            g, new_L, new_M, new_V, action, reward = q_learning_episode(gaze_score, state)
            
            next_gaze_score = next(main_generator) 
            next_gaze_bin = assign_gaze_bin(next_gaze_score)
            next_state = (next_gaze_bin, new_L, new_M, new_V)
            
            print(f"Next gaze score and state: {next_gaze_score}, {next_state}")
            
            if state not in q_table:
                q_table[state] = {} 
            if action not in q_table[state]:
                q_table[state][action] = 0 
                
            # Q-value update
            max_future_q = max(q_table.get(next_state, {}).values(), default=0)
            current_q_value = q_table.get(state, {}).get(action, 0)

            # Update the Q-value using the Q-learning formula
            if state not in q_table:
                q_table[state] = {}
                
            q_table[state][action] = current_q_value + alpha * (reward + gamma * max_future_q - current_q_value)
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            
            step_data = {
                "episode": episode + 1,
                "state": state,
                "action": action,  
                "reward": reward, 
                "next_state": next_state,
                "gaze_score": gaze_score,
                "next_gaze_score": next_gaze_score,
            }
            
            episode_data.append(step_data)
            state = next_state
            previous_state = next_state
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            break 
        save_q_table_episode(episode + 1, q_table)
        save_training_data(episode_data)
        if (episode + 1) % 10 == 0:  
            save_q_table(q_table)
    print("Q-Learning Training Complete.")
    save_q_table(q_table)   
     
def save_q_table_episode(episode, q_table, filename_template="q_table_episode_{episode}.pkl"):
    try:
        filename = os.path.join(SAVE_DIR, filename_template.format(episode=episode))
        with open(filename, 'wb') as f:
            pickle.dump(q_table, f)
        print(f"Q-table for episode {episode} saved to {filename}.")
    except Exception as e:
        print(f"Error saving Q-table for episode {episode}: {e}")
        raise
    
def save_training_data(data, filename="1. training_data.json"):
    try:
        filepath = os.path.join(SAVE_DIR, filename)

        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        if isinstance(data, list):
            existing_data.extend(data)
        else:
            existing_data.append(data)

        with open(filepath, 'w') as f:
            json.dump(existing_data, f, indent=4)

        print(f"Training data saved to {filepath}.")
    except Exception as e:
        print(f"Error saving training data: {e}")
        
def save_q_table(q_table, filename="q_table_full.pkl"):
    try:
        filepath = os.path.join(SAVE_DIR, filename)
        with open(filepath, 'wb') as f: 
            pickle.dump(q_table, f)
        print(f"Full Q-table saved to {filepath}.")
    except Exception as e:
        print(f"Error saving full Q-table: {e}")
        raise       
    
if __name__ == "__main__":
    train_q_learning()
