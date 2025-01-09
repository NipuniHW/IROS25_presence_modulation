import random
import json

states = ["Social", "Alarmed"]
gaze_states = ["Gaze", "No Gaze"]
actions = {
    "Social": ["GestureA", "GestureB", "GestureC"],
    "Alarmed": ["ActionA", "ActionB", "ActionC"]
} 
# GestureA -> More social, GestureB -> Neutral, GestureC -> Silent
# ActionA -> More alarmed, ActionB -> Neutral, ActionC -> Silent
q_table = {
    (state, gaze): {action: 0 for action in actions[state]} 
    for state in states 
    for gaze in gaze_states
}


alpha = 0.1
gamma = 0.9 
epsilon = 0.1
episodes = 10000 

def reward(state, gaze, action):
    if state == "Social" and gaze == "No Gaze":
        return 1 if action == "GestureA" else 0 if action == "GestureB" else -1
    elif state == "Social" and gaze == "Gaze":
        return 1 if action == "GestureB" else 0 if action == "GestureA" else -1 
    elif state == "Alarmed" and gaze == "Gaze":
        return 1 if action == "ActionA" else 0 if action == "ActionB" else -1
    elif state == "Alarmed" and gaze == "No Gaze":
        return 1 if action == "ActionB" else 0 if action == "ActionA" else -1
    else:
        return -1

def train_q_learning():
    global q_table

    for episode in range(episodes):
        
        state = random.choice(states)
        gaze = random.choice(gaze_states)

        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions[state])
        else:
            action = max(q_table[(state, gaze)], key=q_table[(state, gaze)].get)

        r = reward(state, gaze, action)

        next_state = random.choice(states)
        next_gaze = random.choice(gaze_states)

        max_future_q = max(q_table[(next_state, next_gaze)].values())
        q_table[(state, gaze)][action] += alpha * (r + gamma * max_future_q - q_table[(state, gaze)][action])

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} completed.")

    save_q_table()

def save_q_table(filename="q_table.json"):
    string_keyed_q_table = {str(key): value for key, value in q_table.items()}
    with open(filename, "w") as f:
        json.dump(string_keyed_q_table, f)
    print(f"Q-table saved to {filename}")

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
