import random

import numpy as np
from connection import Connection
from mdp_formulation import low_gaze_config

# # Create a proxy to the AL services
# behavior_mng_service = session.service("ALBehaviorManager")
# tts = session.service("ALTextToSpeech")
# leds = session.service("ALLeds")



def update_behavior(action, light, movement, volume):
    l_action, m_action, v_action = action
    
    if l_action == "Increase L":
        light = min(10, light + 1)
    elif l_action == "Decrease L":
        light = max(0, light - 1)
        
    if m_action == "Increase M":
        movement = min(5, movement + 1)
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

# Function to execute an action (this is an example, modify as needed)
def execute_action(light, movement, volume):
    update_lights(light)
    update_movements(movement)
    update_volume(volume)   
    
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
    # LeRobot.behavior_mng_service.stopAllBehaviors()
    LeRobot.behavior_mng_service.startBehavior("attention_actions_2/" + str(movement)) 
 
# To update lights
def update_lights(light):
    global LeRobot
    if light == 0:
        light_n = 0.1
    else:
        light_n = round(max(0, light/10), 1)
    set_all_leds(LeRobot.leds, light_n)    
    
def set_all_leds(leds, light_n):
    for led in low_gaze_config.led_actuators:
        leds.setIntensity(led, light_n)
        