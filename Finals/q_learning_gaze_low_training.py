import random
#from gaze_score import main
# from gaze import main
import json 
# from connection import Connection
# from openai import OpenAI
# from dotenv import load_dotenv
import os
import json
import pickle
import time
from multiprocessing import Process, Queue
import pdb

# # Connect Pepper robot
# pepper = Connection()
# session = pepper.connect('localhost', '33863')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
# # Create a proxy to the AL services
# behavior_mng_service = session.service("ALBehaviorManager")
# tts = session.service("ALTextToSpeech")
# leds = session.service("ALLeds")
        
# Load the environment variables from the .env file
# load_dotenv()

# Initialize OpenAI client
# client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Parameters
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

# Q-Table Initialization
q_table = {}
for gaze_bin in gaze_bins:
            for light in behavior_levels:
                for movement in behavior_levels:
                    for volume in behavior_levels:
                        state = (gaze_bin, light, movement, volume)        # State space: (gaze_bin, lights, movements, volume)
                        q_table[state] = {action: 0 for action in actions}  # Each action gets a Q-value

pdb.set_trace()
#print(q_table)
                    
# Parameters for Q-learning
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
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Explore
    else:
        return max(q_table[state], key=q_table[state].get)  # Exploit

def update_behavior(state, action):
        gaze_bin, light, movement, volume = state
        l_action, m_action, v_action = action
        
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
            light = min(10, volume + 0)
        elif m_action == "Keep M":
            movement = max(0, movement - 0)
        elif v_action == "Keep V":
            volume = max(0, volume - 0)
        
        print(f"L, M, V:", light, movement, volume)
        # if light == 0:
        #     light_n = 0.1
        # else:
        #     light_n = round(max(0, light/10), 1)
        # print(f"Light_n: {light, light_n}")

        
        # leds.setIntensity("Face/Led/Blue/Left/225Deg/Actuator/Value", light_n)
        # leds.setIntensity("Face/Led/Blue/Left/270Deg/Actuator/Value", light_n)            
        # leds.setIntensity("Face/Led/Green/Left/225Deg/Actuator/Value", light_n)
        # leds.setIntensity("Face/Led/Green/Left/270Deg/Actuator/Value", light_n)
        # leds.setIntensity("Face/Led/Red/Left/270Deg/Actuator/Value", light_n)
    
        # behavior_mng_service.stopAllBehaviors()
        # behavior_mng_service.startBehavior("modulated_actions/" + str(movement)) 

        # volume_n = max(0,volume/10)
        # tts.setVolume(volume_n)
        # tts.say("Beep beep")
        
        return (gaze_bin, light, movement, volume)
    
def assign_gaze_bin(gaze_score, num_bins=10):
        if not 0 <= gaze_score <= 100:
            raise ValueError("Gaze score must be between 0 and 100.")

        # Calculate the bin size
        bin_size = 100 / num_bins

        # Determine the bin index (0-indexed)
        gaze_bin = int(gaze_score // bin_size)

        # Handle edge case for a score of 100
        if gaze_bin == num_bins:
            gaze_bin = num_bins - 1

        return gaze_bin
    
def q_learning_episode(gaze_score, state):
    global epsilon
    try:
        gaze_bin, old_light, old_movement, old_volume = state
        print(f"Current state at episode: {state}\n")
        
        expected_min, expected_max = expected_ranges
        print(f"Expected min/ max: {expected_min}, {expected_max} \n")
       
        action = select_action(state)
        print(f"Action selected: {action} \n")
        
        reward = get_reward(gaze_score) 
        if reward is None:
            raise ValueError("Reward is None")

        err_state = (gaze_bin, 2, 2, 2)
        
        if (old_light == 0 and old_movement ==0) or  (old_light == 0 and old_volume ==0) or (old_movement == 0 and old_volume ==0) or (old_movement == 0 and old_volume ==0 and old_light ==0) or (old_light == 10 and old_movement ==10) or  (old_light == 10 and old_volume == 10) or (old_movement == 10 and old_volume == 10) or (old_movement == 10 and old_volume == 10 and old_light == 10) : 
             _, new_light, new_movement, new_volume = update_behavior(err_state, action)
        else:
            _, new_light, new_movement, new_volume = update_behavior(state, action)
            
        print(f"Old state at episode: {gaze_bin}, {old_light}, {old_movement}, {old_volume} \n")
    #    print(f"New state at episode: {gaze_bin}, {new_light}, {new_movement}, {new_volume} \n")

        return gaze_bin, new_light, new_movement, new_volume, action, reward 
    except Exception as e:
        print(f"Error in q_learning_episode: {e}")
        return gaze_bin, new_light, new_movement, new_volume, action, 0 
        
# # Q-learning training function   
# def train_q_learning():
#     global q_table, epsilon
#     print("Starting Q-learning training...")
         
#     main_generator = main()  # Initialize the generator from the main function
#     L, M, V = 0, 0, 0  # Initial behavior levels
#     previous_state = None
#     episode_data = []
    
#     for episode in range(num_episodes):
#         print(f"Episode {episode + 1}/1000")

#         try:
#             gaze_score = next(main_generator)        
#             print(f"Received gaze score: {gaze_score}")

#             gaze_bin = assign_gaze_bin(gaze_score)

#             # Initialize episode data
#             episode_data = []
                
#             # Define the current state
#             if previous_state is None:
#                 state = (gaze_bin, L, M, V)
#             else:
#                 state = previous_state

#             print(f"State at learning: {state}")
        
#             # Perform Q-learning episode
#             g, new_L, new_M, new_V, action, reward = q_learning_episode(gaze_score, state)

#             # Get the next gaze_score from the queue
#             next_gaze_score = next(main_generator)  
            
#             # Assign next gaze bin
#             next_gaze_bin = assign_gaze_bin(next_gaze_score)

#             # Define the next state
#             next_state = (next_gaze_bin, new_L, new_M, new_V)
#             print(f"Next gaze score and state: {next_gaze_score}, {next_state}")

#             # Ensure the state and action exist in the Q-table
#             if state not in q_table:
#                 q_table[state] = {}  # Initialize the state entry

#             if action not in q_table[state]:
#                 q_table[state][action] = 0  # Initialize action value
    
#             # Q-value update
#             max_future_q = max(q_table.get(next_state, {}).values(), default=0)
#             current_q_value = q_table[state][action]
#             q_table[state][action] = current_q_value + alpha * (reward + gamma * max_future_q - current_q_value)

#             # Update epsilon
#             epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
#             # Log the step's data
#             step_data = {
#                 "episode": episode,
#                 "state": state,
#                 "action": action,  
#                 "reward": reward, 
#                 "next_state": next_state,
#                 "gaze_score": gaze_score,
#                 "next_gaze_score": next_gaze_score
#             }
#             episode_data.append(step_data)

#             # Update state and previous_state
#             state = next_state
#             previous_state = next_state

#         except Exception as e:
#             print(f"Error in episode {episode}: {e}")
#             break  # Exit the step loop safely if an error occurs

#         # Save Q-table for the episode
#         save_q_table_episode(episode + 1, q_table)

#         # Save training data for this episode
#         save_training_data(episode_data)

#         # Periodically save the full Q-table
#         if (episode + 1) % 10 == 0:  # Every 10 episodes
#             save_q_table(q_table)

#     print("Q-Learning Training Complete.")me
    
def train_q_learning():
    global q_table, epsilon
    print("Starting exhaustive Q-learning training...")  

    episode_data = []
    #main_generator = main()
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")

        for gaze_bin in gaze_bins:
            for light in behavior_levels:
                for movement in behavior_levels:
                    for volume in behavior_levels:
                        state = (gaze_bin, light, movement, volume)

                        # Iterate through only valid actions based on adjustment rules
                        for action in actions:
                            l_action, m_action, v_action = action
                            #gaze_score = next(main_generator)  
                            gaze_score = gaze_bin * 10
                            expected_min, expected_max = expected_ranges
                            print(f"Expected min/ max: {expected_min}, {expected_max} \n")

                            # Determine adjustment based on gaze score
                            if gaze_score > expected_max:
                                adjustment = -1  # Reduce behavior level
                            elif gaze_score < expected_min:
                                adjustment = 1  # Increase behavior level
                            else:
                                adjustment = 0  # Keep behavior level unchanged
                            
                            print(f"Adjustment: {adjustment} \n")


                            # Ensure action respects the adjustment rule
                            if adjustment == -1 and ("Increase" in action): 
                                continue
                            if adjustment == +1 and ("Decrease" in action): 
                                continue
                            if adjustment == 0 and (l_action != "Keep L" or m_action != "Keep M" or v_action != "Keep V"):
                                continue

                            # Simulate gaze score transition
                            gaze_score = random.randint(0, 100)  # Synthetic gaze score
                            reward = get_reward(gaze_score)
                            _, new_light, new_movement, new_volume = update_behavior(state, action)
                            next_gaze_bin = assign_gaze_bin(random.randint(0, 100))  # Simulated transition
                            next_state = (next_gaze_bin, new_light, new_movement, new_volume)

                            # Q-learning update
                            if state not in q_table:
                                q_table[state] = {}
                            if action not in q_table[state]:
                                q_table[state][action] = 0 

                            max_future_q = max(q_table.get(next_state, {}).values(), default=0)
                            current_q_value = q_table[state][action]
                            q_table[state][action] = current_q_value + alpha * (reward + gamma * max_future_q - current_q_value)

                            # Store episode data
                            step_data = {
                                "episode": episode,
                                "state": state,
                                "action": action,  
                                "reward": reward, 
                                "next_state": next_state,
                                "gaze_score": gaze_score
                            }
                            episode_data.append(step_data)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Save training data periodically
        save_q_table_episode(episode + 1, q_table)
        save_training_data(episode_data)

        if (episode + 1) % 10 == 0:
            save_q_table(q_table)

    print("Exhaustive Q-Learning Training Complete.")
    save_q_table(q_table)


def save_q_table_episode(episode, q_table, filename_template="q_table_episode_{episode}.pkl"):
    try:
        filename = filename_template.format(episode=episode)
        with open(filename, 'wb') as f:
            pickle.dump(q_table, f)
        print(f"Q-table for episode {episode} saved to {filename}.")
    except Exception as e:
        print(f"Error saving Q-table for episode {episode}: {e}")
        raise

def save_training_data(data, filename="training_data.json"):
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        # Flattening the data structure
        if isinstance(data, list):
            existing_data.extend(data)
        else:
            existing_data.append(data)

        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=4)

        print(f"Training data saved to {filename}.")

    except Exception as e:
        print(f"Error saving training data: {e}")


def save_q_table(q_table, filename="q_table_full.pkl"):
    try:
        with open(filename, 'wb') as f:  # 'wb' for writing in binary mode
            pickle.dump(q_table, f)
        print(f"Full Q-table saved to {filename}.")
    except Exception as e:
        print(f"Error saving full Q-table: {e}")
        raise

        
if __name__ == "__main__":
    train_q_learning()
