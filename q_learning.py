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

import numpy as np
import random

# Initialize Q-table
states = ["Social", "Alarmed"]
actions = {
    "Social": ["GestureA", "GestureB", "GestureC"],
    "Alarmed": ["ActionA", "ActionB", "ActionC"]
}
q_table = {state: {action: 0 for action in actions[state]} for state in states}

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Mock reward function
def mock_reward(state, action):
    if state == "Social":
        if action == "GestureA": return 1
        elif action == "GestureB": return 0
        else: return -1
    elif state == "Alarmed":
        if action == "ActionA": return 1
        elif action == "ActionB": return 0
        else: return -1

# Simulate one Q-learning episode
for episode in range(100):
    state = random.choice(states)  # Get current state from context module
    if state not in states:
        continue  # Ignore other states

    # Select action (epsilon-greedy)
    if random.uniform(0, 1) < epsilon:
        action = random.choice(actions[state])  # Explore
    else:
        action = max(q_table[state], key=q_table[state].get)  # Exploit

    # Get mock reward
    reward = mock_reward(state, action)

    # Update Q-value
    max_future_q = max(q_table[state].values())
    q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * max_future_q - q_table[state][action])

    # Print Q-table after updates
    print(f"Episode {episode+1} Q-table:")
    print(q_table)

# Speech generation (mock integration)
def generate_speech(state):
    if state == "Social":
        return "Generate a more engaging social response."
    elif state == "Alarmed":
        return "Generate a more authoritative and urgent evacuation message."

# Action and speech integration
current_state = "Social"  # Example
best_action = max(q_table[current_state], key=q_table[current_state].get)
speech = generate_speech(current_state)
print(f"Best action for {current_state}: {best_action}")
print(f"Generated speech: {speech}")
