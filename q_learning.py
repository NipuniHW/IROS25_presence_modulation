import numpy as np
import random
from gaze import main
import json 
#from pepper_modulation import movement

# Import
#main()

contexts = ["Disengaged", "Social", "Alarmed"]
expected_ranges = {
    "Disengaged": (0, 30),
    "Social": (31, 60),
    "Alarmed": (61, 100)
}

behaviors = ["Lights", "Movements", "Volume"]
behavior_levels = list(range(11)) 

# Q-Table initialization
q_table = {}
for context in contexts:
    for light in behavior_levels:
        for movement in behavior_levels:
            for volume in behavior_levels:
                state = (context, light, movement, volume)
                q_table[state] = {action: 0 for action in behaviors}

# Parameters for Q-learning
alpha = 0.1  
gamma = 0.9  
epsilon = 0.9  
epsilon_decay = 0.99
min_epsilon = 0.1

def get_reward(context, gaze_score):
    expected_min, expected_max = expected_ranges[context]
    expected_center = (expected_min + expected_max) / 2
    expected_range_width = expected_max - expected_min
    reward = -((abs(gaze_score - expected_center) / (expected_range_width / 2)) ** 2)
    return reward

def select_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(behaviors)  
    else:
        return max(q_table[state], key=q_table[state].get)  

def update_behavior(state, action, adjustment):
    context, light, movement, volume = state
    if action == "Lights":
        light = max(0, min(10, light + adjustment))
    elif action == "Movements":
        movement = max(0, min(10, movement + adjustment))
    elif action == "Volume":
        volume = max(0, min(10, volume + adjustment))
    return (context, light, movement, volume)

def q_learning_episode(context, gaze_score, state):
    global epsilon

    action = select_action(state)
    
    expected_min, expected_max = expected_ranges[context]

    if gaze_score > expected_max:
        adjustment = -1  # reduce
    elif gaze_score < expected_min:
        adjustment = 1  # increase 
    else:
        adjustment = 0  # same
        
    new_state = update_behavior(state, action, adjustment)

#    new_gaze_score = simulate_gaze_feedback(new_state) 
    reward = get_reward(context, gaze_score)

    # Q-value update
    max_future_q = max(q_table[new_state].values())
    q_table[state][action] += alpha * (reward + gamma * max_future_q - q_table[state][action])

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return new_state, gaze_score

# Q-learning training function
def train_q_learning():
    global q_table

    # Training Loop
    for episode in range(1000): 
        context = random.choice(contexts)
        gaze_score = gaze_score
        state = (context, 5, 5, 5)  

        for step in range(10):  # Limit steps per episode
            state, gaze_score = q_learning_episode(context, gaze_score, state)

    print("Q-Learning Training Complete.")

    save_q_table()
    
# Save Q-table
def save_q_table(filename="q_table.json"):

    string_keyed_q_table = {str(key): value for key, value in q_table.items()}
    with open(filename, "w") as f:
        json.dump(string_keyed_q_table, f)
    print(f"Q-table saved to {filename}")

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