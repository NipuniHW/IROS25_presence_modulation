import numpy as np
import random
from gaze import main
import json 
from connection import Connection
from openai import OpenAI
from dotenv import load_dotenv
import qi
import os

# Connect Pepper robot
pepper = Connection()
session = pepper.connect('localhost', '38965')

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
gaze_ranges = list(range(0, 101, 10)) 
expected_ranges = {
    "Disengaged": (0, 30),
    "Social": (31, 60),
    "Alarmed": (61, 100)
}

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

#behaviors = ["Lights", "Movements", "Volume"]
behavior_levels = list(range(11))  # Levels from 0 to 10

# Q-Table Initialization
q_table = {}
for context in contexts:
    for gaze in gaze_ranges:
        for light in behavior_levels:
            for movement in behavior_levels:
                for volume in behavior_levels:
                        state = (context, gaze, light, movement, volume)        # State space: (context, gaze_score, lights, movements, volume)
                        q_table[state] = {action: 0 for action in actions}
                    
# Parameters for Q-learning
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.9  # Initial exploration rate
epsilon_decay = 0.99
min_epsilon = 0.1
num_episodes = 1000

def get_reward(context, gaze_score):
    expected_min, expected_max = expected_ranges[context]
    expected_center = (expected_min + expected_max) / 2
    expected_range_width = expected_max - expected_min
    reward = -((abs(gaze_score - expected_center) / (expected_range_width / 2)) ** 2)
    return reward

def select_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Explore
    else:
        return max(q_table[state], key=q_table[state].get)  # Exploit

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

def update_behavior(state, action, adjustment, prompt_text):
    context, gaze, light, movement, volume = state
    l_action, m_action, v_action = action
    
    if l_action == "Increase L":
        light = min(10, light + adjustment)
    elif l_action == "Decrease L":
        light = max(0, light + adjustment)
    
    if m_action == "Increase M":
        movement = min(10, movement + adjustment)
    elif m_action == "Decrease M":
        movement = max(0, movement + adjustment)
        
    if v_action == "Increase V":
        volume = min(10, volume + adjustment)
    elif v_action == "Decrease V":
        volume = max(0, volume + adjustment)
        
    light_n = light/10
    leds.setIntensity("Face/Led/Blue/Left/225Deg/Actuator/Value", light_n)
    leds.setIntensity("Face/Led/Blue/Left/270Deg/Actuator/Value", light_n)            
    leds.setIntensity("Face/Led/Green/Left/225Deg/Actuator/Value", light_n)
    leds.setIntensity("Face/Led/Green/Left/270Deg/Actuator/Value", light_n)
    leds.setIntensity("Face/Led/Red/Left/270Deg/Actuator/Value", light_n)
   
    behavior_mng_service.stopAllBehaviors()
    behavior_mng_service.startBehavior("modulated_actions/" + movement) 

    volume_n = volume/10
    tts.setVolume(volume_n)
    tts.say(prompt_text)
    
    return (context, light, movement, volume)

def q_learning_episode(context, gaze_score, transcription, state):
    global epsilon

    action = select_action(state)
    
    expected_min, expected_max = expected_ranges[context]
    
    # Determine adjustment based on gaze score
    if gaze_score > expected_max:
        adjustment = -1  # Reduce behavior level
    elif gaze_score < expected_min:
        adjustment = 1  # Increase behavior level
    else:
        adjustment = 0  # Keep behavior level unchanged
    
    prompt_text = generate_gpt_prompt(context, transcription)    
    new_state = update_behavior(state, action, adjustment, prompt_text)

#    new_gaze_score = simulate_gaze_feedback(new_state) 
    reward = get_reward(context, gaze_score)

    # Q-value update
    max_future_q = max(q_table[new_state].values())
    q_table[state][action] += alpha * (reward + gamma * max_future_q - q_table[state][action])

    # Update epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return new_state, gaze_score

# Q-learning training function   
def train_q_learning():
    global q_table
    load_q_table()
    
    # Training Loop
    print("Starting Q-learning training...")
    main_generator = main()  # Initialize the generator from the main function

    previous_state = None
    
    for episode in range(1000):
        print(f"Episode {episode + 1}/1000")
        gaze_score, context, transcription = next(main_generator)
        print(f"Received gaze score: {gaze_score}, Context: {context}, Trasncription : {transcription}")

        # Initialize the state if it's the first episode
        if previous_state is None:
            state = (context, gaze_score, 5, 5, 5)  # Assuming initial behavior levels as 5 for light, movement, and volume
        else:
            state = previous_state

        print(f"State at learning: {state}")
        
        for step in range(10):  # Limit steps per episode
            try:
                # Perform a Q-learning episode step
                state = q_learning_episode(context, gaze_score, transcription, state)
                
                # Update the previous state
                previous_state = state
                
            except Exception as e:
                print(f"Error during Q-learning episode: {e}")
                break  # Exit step loop safely if an error occurs

        # Save progress every few episodes
        if episode % 10 == 0:
            print(f"Saving Q-table at episode {episode}")
            save_q_table() 

    print("Q-Learning Training Complete.")
    save_q_table()
       
# Save Q-table
def save_q_table(filename="q_table.json"):
    try:
        string_keyed_q_table = {str(key): value for key, value in q_table.items()}
        with open(filename, "w") as f:
            json.dump(string_keyed_q_table, f, indent=4)
        print(f"Q-table saved to {filename}")
    except Exception as e:
            print(f"Error saving Q-table: {e}")

# Load Q-table
def load_q_table(filename="q_table.json"):
    global q_table
    try:
        with open(filename, "r") as f:
            string_keyed_q_table = json.load(f)

        q_table = {eval(key): value for key, value in string_keyed_q_table.items()}
        print(f"Q-table loaded from {filename}")
    except FileNotFoundError:
        print("No saved Q-table found. Starting fresh.")
        
if __name__ == "__main__":
    train_q_learning()
