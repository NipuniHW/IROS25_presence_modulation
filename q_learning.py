# MIT License
# 
# Copyright (c) [2024] Modulate Presence
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#import numpy as np
import random
from presence_modulation import listen_and_process 
import json

# Initialize Q-table
states = ["Social", "Alarmed"]
gaze = ["Gaze", "No Gaze"]
actions = {
    "Social": ["GestureA", "GestureB", "GestureC"],
    "Alarmed": ["ActionA", "ActionB", "ActionC"]
} # GestureA -> More social, GestureB -> Neutral, GestureC -> Silent //// ActionA -> More alarmed, ActionB -> Neutral, ActionC -> Silent
q_table = {state: {action: 0 for action in actions[state]} for state in states}

alpha = 0.1  # Learning rate
gamma = 0.9  # Discount 
epsilon = 0.1  # Exploration 
episodes = 1000 # No of episodes

# Reward function #Add the Gaze function here
def reward(state, gaze, action):
    if state == "Social" and gaze == "No Gaze":
        if action == "GestureA": return 1
        elif action == "GestureB": return 0
        else: return -1
    elif state == "Alarmed" and gaze == "Gaze":
        if action == "ActionA": return 1
        elif action == "ActionB": return 0
        else: return -1

# Training q-learning
def train_q_learning():
    global q_table
    
    for episode in range(episodes):
        
        state = random.choice(states)

        # Select action (epsilon-greedy)
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions[state])  # Explore
        else:
            action = max(q_table[state], key=q_table[state].get)  # Exploit

        # Get reward
        reward = reward(state, action)

        # Update Q-value
        max_future_q = max(q_table[state].values())
        q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * max_future_q - q_table[state][action])

        # Print Q-table after updates
        print(f"Episode {episode+1} Q-table:")
        print(q_table)
    
    # Save the trained Q-table
    save_q_table()

# Save Q-table
def save_q_table(filename="q_table.json"):
    with open(filename, "w") as f:
        json.dump(q_table, f)
    print("Q-table saved to", filename)

# Load Q-table
def load_q_table(filename="q_table.json"):
    global q_table
    try:
        with open(filename, "r") as f:
            q_table = json.load(f)
        print("Q-table loaded from", filename)
    except FileNotFoundError:
        print("No saved Q-table found. Starting fresh.")
        
# Speech generation (integration)
def generate_speech(state):
    if state == "Social":
        return "Generate a more engaging social response."
    elif state == "Alarmed":
        return "Generate a more authoritative and urgent evacuation message."

# Real-time Q-learning integration
def q_learning_pipeline(final_label):
    if final_label not in states:
        print(f"State '{final_label}' ignored (not part of learning states).")
        return

    # Select action (epsilon-greedy)
    if random.uniform(0, 1) < epsilon:
        action = random.choice(actions[final_label])  # Explore
    else:
        action = max(q_table[final_label], key=q_table[final_label].get)  # Exploit

    # Reward for real-time demonstration
    reward = reward(final_label, action)

    # Simulate transition to next state (mock, replace with real logic)
    next_state = random.choice(states)

    # Update Q-value
    max_future_q = max(q_table[next_state].values())
    q_table[final_label][action] = q_table[final_label][action] + alpha * (reward + gamma * max_future_q - q_table[final_label][action])

    # Perform the chosen action (mock, replace with actual Pepper behavior)
    print(f"Performing action '{action}' for state '{final_label}'.")

    # Save updated Q-table 
    save_q_table()

# Run real-time integration
def run_real_time_q_learning():
    load_q_table()  # Load existing Q-table 
    while True:

        _, final_label = listen_and_process()  

        # Use Q-learning pipeline
        q_learning_pipeline(final_label)

        # exit condition
        user_input = input("Press 'q' to quit or any other key to continue: ").lower()
        if user_input == 'q':
            break
