import random
import sys
import cv2
import numpy as np
import time
import pandas as pd
import argparse
from pepper import Pepper
# from gaze import main
from gaze_interface_controller import main
from connection import Connection
import qi
import threading
from mdp_formulation import low_gaze_config

def load_q_table(file_path):
    q_table = pd.read_csv(file_path)
    q_table.index = q_table.index.astype(int)
    q_table.set_index(q_table.columns[0], inplace=True)
    q_table.index.name = "State"
    return q_table

# To update lights
def update_lights(light):
    global LeRobot
    if light == 0:
        light_n = 0.1
    else:
        light_n = round(max(0, light/10), 1)
    set_all_leds(LeRobot.leds, light_n)    
    
def set_all_leds(leds, light_n):
    # led_actuators = low_gaze_config.led_actuators

    for led in low_gaze_config.led_actuators:
        leds.setIntensity(led, light_n)
    
# To update volume
def update_volume(volume):    
    global LeRobot
    volume_n = round(max(0, volume/10), 1)
    # print(f"Volume_n: {volume, volume_n}")
    LeRobot.tts.setVolume(volume_n)
    
    # List of random greetings or catchphrases
    greetings = [
        "Hello there!",
        "How's it going?",
        "Nice to see you!",
        "What's up?",
        "Greetings!",
        "Hey, how are you?",
        "Excuse me",
        "Good day!",
        "Hi there!",
        "Howdy!",
        "Welcome!",
        "beep boop beep",
        "I am here!",
        "Hello, human!",
        "beep beep beep" # just for Damith
    ]    
    # Randomly pick a greeting
    random_greeting = random.choice(greetings)
    LeRobot.tts.say(random_greeting)

# To update movements
def update_movements(movement):
    global LeRobot
    LeRobot.behavior_mng_service.stopAllBehaviors()
    LeRobot.behavior_mng_service.startBehavior("modulated_actions/" + str(movement)) 

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

def test_q_learning(q_table_path, duration_minutes=2):
    q_table = load_q_table(q_table_path)
    gaze_generator = main()   
    light, movement, volume = 0, 0, 0  # Default values 

    def process_q_learning(gaze_score):
        state = get_gaze_bin(gaze_score)
        print(f"Current state: {state}")
        action = choose_action(state, q_table)
        print(f"Chosen action: {action}")
        nonlocal light, movement, volume
        light, movement, volume = update_behavior(action, light, movement, volume)

    start_time = time.time()
    interval = 2  # Interval in seconds
    end_time = start_time + duration_minutes * 60  # Calculate end time
    gaze_score_average_vector = []
    
    while time.time() < end_time:
        gaze_score = next(gaze_generator)
        gaze_score_average_vector.append(gaze_score)
        print("added the number ", gaze_score) 
        
        current_time = time.time()
        
        if current_time - start_time >= interval:
            gaze_score_average = sum(gaze_score_average_vector) / len(gaze_score_average_vector)
            process_q_learning(gaze_score_average)
            gaze_score_average_vector = []
            print("########average gaze score ", gaze_score_average)
            print("########updated the behavior")
            # process_q_learning(gaze_score)
            start_time = current_time  # Reset the timer

        # process_q_learning(gaze_score)
        
        # Delay the loop by 180ms
        time.sleep(0.18)

    print("Test completed")

if __name__ == "__main__":    
    global LeRobot
    LeRobot = Pepper()
    try:
        LeRobot.connect("pepper.local", 9559)
        if not LeRobot.is_connected:
            sys.exit(1)
            
        parser = argparse.ArgumentParser(description='Q-Learning Testing')
        parser.add_argument('--q_table', type=str, required=True, help='Path to the Q-table CSV file')
        args = parser.parse_args()

        test_q_learning(args.q_table)
        del LeRobot 
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Cleaning up...")
        del LeRobot   
        cv2.destroyAllWindows()
        sys.exit(0)