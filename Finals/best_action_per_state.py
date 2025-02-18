import pandas as pd

# Load the dataset
file_path = "/home/nipuni/Documents/IROS25_presence_modulation/Finals/test_data_low/test_data_low_episode_6.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# Identify the highest action for each state
df.set_index("State", inplace=True)
highest_actions = df.idxmax(axis=1)  # Get action with the highest Q-value per state

# Convert to a readable format
highest_actions = highest_actions.reset_index()
highest_actions.columns = ["State", "Best Action"]

# Display the results
print(highest_actions)
