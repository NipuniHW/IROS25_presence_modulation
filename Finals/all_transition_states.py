import json

class QLearningAgent:
    def __init__(self):
        # Define expected gaze score range
        self.expected_ranges = (0, 30)
        
        # Define all possible actions
        self.actions = [
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
        
        # Define gaze bins and behavior levels
        self.gaze_bins = list(range(1, 11))  # Gaze score bins (1-10)
        self.behavior_levels = list(range(11))  # Levels for L, M, V (0-10)
        
        # Initialize Q-table
        self.q_table = {}
        for gaze_bin in self.gaze_bins:
            for light in self.behavior_levels:
                for movement in self.behavior_levels:
                    for volume in self.behavior_levels:
                        state = (gaze_bin, light, movement, volume)  # State space: (gaze_bin, L, M, V)
                        self.q_table[state] = {action: 0 for action in self.actions}  # Initialize Q-values for all actions

    def get_next_state(self, current_state, action, next_gaze_bin):
        """
        Compute the next state based on the current state, action, and next gaze bin.
        """
        gaze_bin, light, movement, volume = current_state
        
        # Apply action to L, M, V
        if "Increase L" in action:
            light = min(light + 1, 10)
        elif "Decrease L" in action:
            light = max(light - 1, 0)
        
        if "Increase M" in action:
            movement = min(movement + 1, 10)
        elif "Decrease M" in action:
            movement = max(movement - 1, 0)
        
        if "Increase V" in action:
            volume = min(volume + 1, 10)
        elif "Decrease V" in action:
            volume = max(volume - 1, 0)
        
        # Use the provided next_gaze_bin
        return (next_gaze_bin, light, movement, volume)

    def get_possible_transitions(self):
        """
        Generate all possible transitions: (current_state, action_taken, next_state).
        """
        transitions = []
        for state in self.q_table:
            for action in self.actions:
                for next_gaze_bin in self.gaze_bins:  # Iterate over all possible gaze bins
                    next_state = self.get_next_state(state, action, next_gaze_bin)
                    transitions.append({
                        "current_state": state,
                        "action_taken": action,
                        "next_state": next_state
                    })
        return transitions

# Initialize the Q-learning agent
agent = QLearningAgent()

# Get all possible transitions
transitions = agent.get_possible_transitions()

# Save transitions to a JSON file
output_file = "transitions.json"
with open(output_file, "w") as f:
    json.dump(transitions, f, indent=4)

print(f"All possible transitions have been saved to {output_file}.")