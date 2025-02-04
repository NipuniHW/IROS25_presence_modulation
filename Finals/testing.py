import numpy as np
import time
#from pepper_modulation import update_lights, update_volume, update_movements
import pandas as pd
from mdp_formulation import low_gaze_config, medium_gaze_config, high_gaze_config
from gaze import main
from connection import Connection
import qi

# Connect Pepper robot
pepper = Connection()
# session = pepper.connect('pepper.local', '9559')
# session = pepper.connect('127.0.0.1', '39603')
session = pepper.connect('localhost', '35501')

# Create a proxy to the AL services
behavior_mng_service = session.service("ALBehaviorManager")
tts = session.service("ALTextToSpeech")
leds = session.service("ALLeds")
    
# Load the trained Q-table
q_table = pd.read_csv("/home/nipuni/Documents/IROS25_presence_modulation/Finals/table_high.csv")

# Ensure the first column is treated as the state index
# Ensure the index is treated as integers (in case they were read as strings)
q_table.index = q_table.index.astype(int)
q_table.set_index(q_table.columns[0], inplace=True)
q_table.index.name = "State"  


config = high_gaze_config

# Define the state and action space
state_space = config.states
action_space = config.actions  

# To update lights
def update_lights(light):
    if light == 0:
        light_n = 0.1
    else:
        light_n = round(max(0, light/10), 1)
        
    # print(f"Light_n: {light, light_n}")
    leds.setIntensity("Face/Led/Blue/Left/225Deg/Actuator/Value", light_n)
    leds.setIntensity("Face/Led/Blue/Left/270Deg/Actuator/Value", light_n)            
    leds.setIntensity("Face/Led/Green/Left/225Deg/Actuator/Value", light_n)
    leds.setIntensity("Face/Led/Green/Left/270Deg/Actuator/Value", light_n)
    leds.setIntensity("Face/Led/Red/Left/270Deg/Actuator/Value", light_n)
    
# To update volume
def update_volume(volume):    
    volume_n = round(max(0, volume/10), 1)
    print(f"Volume_n: {volume, volume_n}")
    tts.setVolume(volume_n)
    tts.say("Beep boop beep boop")

# To update movements
def update_movements(movement):
    behavior_mng_service.stopAllBehaviors()
    behavior_mng_service.startBehavior("attention_actions/" + str(movement)) 

def get_gaze_bin(gaze_score):
    if gaze_score < 0.0 or gaze_score > 100.0:
        raise ValueError("Raw gaze score must be between 0.0 and 100.0")

    if gaze_score <= 30.0:
        return int((gaze_score / 30.0) * 3)  # Scale 0-30 to 0-3
    elif gaze_score <= 60.0:
        return int(4 + ((gaze_score - 31.0) / 29.0) * 2)  # Scale 31-60 to 4-6
    else:
        return int(7 + ((gaze_score - 61.0) / 39.0) * 3)  # Scale 61-100 to 7-10

# Function to execute an action (this is an example, modify as needed)
def execute_action(light, movement, volume):
    update_lights(light)
    update_movements(movement)
    update_volume(volume) 

# Function to choose an action based on the current state
def choose_action(state, q_table):
    if state not in q_table.index:
        raise ValueError(f"State {state} is not in the Q-table. Available states: {q_table.index.tolist()}")

    action_values = q_table.loc[state]
    max_q_value = action_values.max()
    best_actions = action_values[action_values == max_q_value].index.tolist()

    chosen_action = np.random.choice(best_actions)  # Randomly select among best actions if tie

    # Ensure the chosen action is split into a list of three components
    return chosen_action.split(", ")  # Split by ", " to separate the values


def update_behavior(action, light, movement, volume):
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
        light = light
    elif m_action == "Keep M":
        movement = movement
    elif v_action == "Keep V":
        volume = volume
    
    print(f"Light: {light}, Movement: {movement}, Volume: {volume}")
    
    # Perform the action
    execute_action(light, movement, volume)
    return light, movement, volume

# Main testing loop
def test_q_learning():
    gaze_generator = main()   
    light, movement, volume = 0, 0, 0  # Default values 
        
    for _ in range(100):  # Test for 100 steps
        gaze_score = next(gaze_generator)
        print(f"Current gaze score: {gaze_score}")
        state = get_gaze_bin(gaze_score)
        print(f"Current state: {state}")
        action = choose_action(state, q_table)
        print(f"Chosen action: {action}")
        light, movement, volume = update_behavior(action, light, movement, volume)

        time.sleep(0.1)
    print("Test completed")

if __name__ == "__main__":
    test_q_learning()