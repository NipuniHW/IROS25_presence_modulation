import numpy as np
import random
from gaze import main
import json 
from connection import Connection
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import pickle

# Connect Pepper robot
pepper = Connection()
session = pepper.connect('localhost', '36123')

# Create a proxy to the AL services
behavior_mng_service = session.service("ALBehaviorManager")
tts = session.service("ALTextToSpeech")
leds = session.service("ALLeds")
        
# Load the environment variables from the .env file
load_dotenv()

# Initialize OpenAI client
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Parameters
contexts = ["Disengaged", "Social", "Alarmed"]
expected_ranges = {
    "Disengaged": (0, 30),
    "Social": (31, 60),
    "Alarmed": (61, 100)
}
gaze_bins = list(range(1, 11))
behaviors = ["Lights", "Movements", "Volume"]
behavior_levels = list(range(11)) 
num_bins = 10
ep_num = 1000

# Q-Table initialization
q_table = {}

for context in contexts:
    for gaze_bin in gaze_bins:
        q_table[(context, gaze_bin)] = {
            (l, m, v): 0 for l in behavior_levels for m in behavior_levels for v in behavior_levels
        }
                    
# Parameters
alpha = 0.1  
gamma = 0.9  
epsilon = 0.9  
epsilon_decay = 0.99
min_epsilon = 0.1

def get_reward(context, gaze_score):
    expected_min, expected_max = expected_ranges[context]
    expected_center = (expected_min + expected_max) / 2
    expected_range_width = expected_max - expected_min
    reward = 1 - (abs(gaze_score - expected_center) / (expected_range_width / 2)) ** 2

    return reward

def select_action(state):
    context, gaze_bin, light, movement, volume = state
    best_action = None
    max_q_value = float('-inf')

    if random.uniform(0, 1) < epsilon:
        # Exploration
        best_action = (
            random.choice(behavior_levels),
            random.choice(behavior_levels), 
            random.choice(behavior_levels),
        )
    else:
        # Exploitation
        for l in behavior_levels:
            for m in behavior_levels:
                for v in behavior_levels:
                    possible_state = (context, gaze_bin, l, m, v)
                    q_value = q_table[(context, gaze_bin)].get((l, m, v), 0) 
                    print(f"q_value: {q_value}, max_q_value: {max_q_value}")
                    if isinstance(q_value, dict):
                        raise ValueError(f"q_table.get returned a dict! Possible state: {possible_state}")
                    if q_value > max_q_value:
                        max_q_value = q_value
                        best_action = (l, m, v)
    
    return best_action

# Function to generate the prompt for Pepper
def generate_gpt_prompt(final_label, transcription):
    if final_label == "Alarmed":
        messages = 'You are Pepper, an interactive agent who will inform on an emegency situation. Generate a clear and firm response for an emergency scenario. Maintain authority while providing reassurance and instructions to help users act safely. You use short sentences. You use maximum of 2 sentences.'
    elif final_label == "Social":
        messages = 'You are Pepper, an interactive friendly agent who is chatty and loves to engage in casual conversations. Do not say who you are except for the name. Do not say "as an AI". You use short sentences. You use maximum of 2 sentences. Keep it engaging but balanced, showing interest and attentiveness without being overbearing.'
    elif final_label == "Disengaged":
        messages = 'Use 0 words.'
      
    # Call the OpenAI API to generate the appropriate response
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": messages}, 
            {"role": "user", "content": transcription}
        ]
    )
    
    # Extract the GPT-generated response
    generated_prompt = response.choices[0].message.content
    print(f"GPT-generated prompt: {generated_prompt}")
    
    return generated_prompt

def update_behavior(state, action, prompt_text):
    context, gaze_bin, light, movement, volume = state
    UL, UM, UV = action
    
    UL = int(UL)
    UM = int(UM)
    UV = int(UV)
        
    light = max(0, min(10, UL))
    if light == 0:
        light_n = 0.1
    else:
        light_n = round(max(0, light/10), 1)
    print(f"Light_n: {light, light_n}")
    leds.setIntensity("Face/Led/Blue/Left/225Deg/Actuator/Value", light_n)
    leds.setIntensity("Face/Led/Blue/Left/270Deg/Actuator/Value", light_n)            
    leds.setIntensity("Face/Led/Green/Left/225Deg/Actuator/Value", light_n)
    leds.setIntensity("Face/Led/Green/Left/270Deg/Actuator/Value", light_n)
    leds.setIntensity("Face/Led/Red/Left/270Deg/Actuator/Value", light_n)
    
    movement = max(0, min(10, UM))
    behavior_mng_service.stopAllBehaviors()
    behavior_mng_service.startBehavior("modulated_actions/" + str(movement)) 
    
    volume = max(0, min(10, UV))
    volume_n = round(max(0, volume/10), 1)
    print(f"Volume_n: {volume, volume_n}")
    tts.setVolume(volume_n)
    tts.say(prompt_text)
    
    return (context, gaze_bin, light, movement, volume)

def q_learning_episode(context, gaze_score, transcription, state):
    global epsilon
    print(f"Current state at episode: {state}\n")
    gaze_bin = assign_gaze_bin(gaze_score)

    if (context, gaze_bin) not in q_table:
        q_table[(context, gaze_bin)] = {}

    if state not in q_table[(context, gaze_bin)]:
        q_table[(context, gaze_bin)][state] = {
            (l, m, v): 0 for l in behavior_levels for m in behavior_levels for v in behavior_levels
        }        
    action = select_action(state)
    print(f"Action selected at episode: {action} \n")
        
    expected_min, expected_max = expected_ranges[context]
    print(f"Expected min/ max: {expected_min}, {expected_max} \n")

    reward = get_reward(context, gaze_score)
    
    prompt_text = generate_gpt_prompt(context, transcription)    
    new_state = update_behavior(state, action, prompt_text)
    print(f"New state at episode: {new_state}\n")
    
    if (context, gaze_bin) not in q_table:
        q_table[(context, gaze_bin)] = {}

    if new_state not in q_table[(context, gaze_bin)]:
        q_table[(context, gaze_bin)][new_state] = {
            (l, m, v): 0 for l in behavior_levels for m in behavior_levels for v in behavior_levels
        }   

    max_future_q = max(q_table[(context, gaze_bin)].get(new_state, {}).values(), default=0)
    current_q_value = q_table[(context, gaze_bin)].get(state, {}).get(action, 0)
    q_table[(context, gaze_bin)][state][action] += alpha * (reward + gamma * max_future_q - current_q_value)

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return new_state

def assign_gaze_bin(gaze_score, num_bins=10):
    if not 0 <= gaze_score <= 100:
        raise ValueError("Gaze score must be between 0 and 100.")
    bin_size = 100 / num_bins
    gaze_bin = int(gaze_score // bin_size)
    if gaze_bin == num_bins:
        gaze_bin = num_bins - 1

    return gaze_bin

def train_q_learning():
    global q_table
    print("Starting Q-learning training...")    
    main_generator = main()  
    L, M, V = 2, 2, 2  # Initial behavior levels
    previous_state = None
    episode_data = []
    
    for episode in range(ep_num):
        print(f"Episode {episode + 1}/1000")
        try:
            # Get the latest gaze score and context from the main generator
            gaze_score, context = next(main_generator)
            transcription = "Hi Pepper"
            print(f"Received gaze score: {gaze_score}, Context: {context}, Transcription: {transcription}")

            gaze_bin = assign_gaze_bin(gaze_score)
            print(f"Assigned Gaze Bin: {gaze_bin}")

            if previous_state is None:
                state = (context, gaze_bin, L, M, V)
            else:
                c, g, prev_L, prev_M, prev_V = previous_state
                state = (context, gaze_bin, prev_L, prev_M, prev_V)

            print(f"State at learning: {state}")
        
            next_state = q_learning_episode(context, gaze_score, transcription, state)
            
            previous_state = next_state
            cn, gb, L, M, V = next_state
            print(f"Next state at learning: {next_state}")
                
            step_data = {
                    "state": state,
                    "action": (L, M, V),
                    "reward": get_reward(context, gaze_score),  
                    "next_state": next_state,
                    "gaze_score": gaze_score,
                    "gaze_bin": gaze_bin,
                    "context": context,
            }
            episode_data.append(step_data)
                
            # Save Q-table at each episode
            save_q_table_episode(q_table, episode)

            # Save training data at each episode
            save_training_data(step_data, episode)

            # Periodically save Q-table
            if (episode + 1) % 10 == 0:
                save_q_table_full(q_table)

        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            break 

    # Save final Q-table and training data after the loop
    print("Q-Learning Training Complete.")
    save_q_table_full(q_table)
    save_training_data(episode_data, ep_num)  # Save all data from the final episode

def save_q_table_full(q_table, filename="q_table_full.pkl"):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(q_table, f)
        print(f"Full Q-table saved to {filename}.")
    except Exception as e:
        print(f"Error saving full Q-table: {e}")
        raise

def save_q_table_episode(q_table, episode, filename_template="q_table_episode_{episode}.pkl"):
    try:
        filename = filename_template.format(episode=episode)
        with open(filename, 'wb') as f:
            pickle.dump(q_table, f)
        print(f"Q-table for episode {episode} saved to {filename}.")
    except Exception as e:
        print(f"Error saving Q-table for episode {episode}: {e}")
        raise

def save_training_data(training_data, episode):
    try:
        filename = f"training_data_episode_{episode}.pkl"  
        with open(filename, 'wb') as f:
            pickle.dump(training_data, f)
        print(f"Training data for episode {episode} saved.")
    except Exception as e:
        print(f"Error saving training data for episode {episode}: {e}")
      
if __name__ == "__main__":
    train_q_learning()
