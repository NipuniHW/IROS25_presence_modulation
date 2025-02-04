from mdp_formulation import GazeFormulationBaseClass, low_gaze_config
import random
import csv
from rewards import low_gaze_reward

def calculate_q_value(q_table, current_state, current_action, reward, next_state_key, config):
    """
    Update the Q-table using the Q-learning formula.
    """
    if next_state_key not in q_table:  # Ensure next state exists in Q-table
        q_table[next_state_key] = {action: 0.0 for action in config.actions.keys()}

    max_future_q = max(q_table[next_state_key].values())  # Best possible Q-value for next state
    q_current = q_table[current_state][current_action]  # Current Q-value
    
    # Q-learning update rule
    q_new = q_current + config.learning_rate * (reward + config.gamma * max_future_q - q_current)
    q_table[current_state][current_action] = q_new  
    
    return q_new

def get_next_state(current_state_key, actions, states):
    """
    Pick a random action and compute the next state.
    """
    chosen_action_key = random.choice(list(actions.keys()))  # Random action selection
    action_vector = actions[chosen_action_key]

    current_gaze = states[current_state_key]
    next_gaze = current_gaze + sum(action_vector)  # Compute new gaze score

    next_gaze = max(0, min(10, next_gaze))  # Ensure within bounds

    # Find the closest matching state key
    next_state_key = min(states.keys(), key=lambda k: abs(states[k] - next_gaze))

    return next_state_key, chosen_action_key

def save_q_table_to_csv(q_table, filename):
    """
    Save the Q-table to a CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['State'] + list(next(iter(q_table.values())).keys())  # Get action headers
        writer.writerow(header)
        
        for state, actions in q_table.items():
            row = [state] + list(actions.values())
            writer.writerow(row)
    
    print(f"Q-table saved to {filename}")

def train_q_learning(q_table, states, actions, reward_function, episodes=1000, epsilon=0.1):
    """
    Train the Q-learning model.
    """
    state_visits = {s: 0 for s in states.keys()}  # Track state visit counts

    for episode in range(episodes):
        for current_state in states.keys():
            next_state_key, chosen_action_key = get_next_state(current_state, actions, states)  # Get next state

            # Compute reward
            reward = reward_function(states[current_state], actions[chosen_action_key])

            # Update Q-value
            calculate_q_value(q_table, current_state, chosen_action_key, reward, next_state_key, low_gaze_config)

            # Track visits
            state_visits[current_state] += 1

    print("State Visit Counts:", state_visits)

if __name__ == "__main__":
    # Load configuration
    config = low_gaze_config
    q_table = {state_key: {action_key: 0.0 for action_key in config.actions.keys()} for state_key in config.states.keys()}

    # Train the model
    train_q_learning(q_table, config.states, config.actions, low_gaze_reward, config.episodes, config.epsilon)

    # Save Q-table
    save_q_table_to_csv(q_table, 'q_table.csv')
