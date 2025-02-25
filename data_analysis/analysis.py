import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import yaml
import glob

# Define experiment condition
condition = 'high'  # Change as needed

# Get all YAML files
yaml_files = glob.glob("/home/nipuni/Documents/IROS25_presence_modulation/day2_exp_data/High/*.yaml")  # Change path as needed

if not yaml_files:
    print("No YAML files found. Check the path!")

colors = ['b', 'g', 'c', 'm', 'y', 'k']
all_gaze_data = {}

plt.figure(figsize=(10, 6))

for idx, file in enumerate(yaml_files):
    print(f"Processing: {file}")
    
    try:
        with open(file, 'r') as f:
            data = yaml.safe_load(f)

        # Extract gaze scores
        timestamps = []
        gaze_states = []
        
        for key, value in data.items():
            if condition in ['low', 'high'] and key.startswith("gaze_state_timestep_"):
                timestamp = int(key.split("_")[-1]) * 3  # Convert index to time (3s intervals)
                timestamps.append(timestamp)
                gaze_states.append(value)   
            elif condition == 'dynamic' and key.startswith("state_subject_1_timestep_"):
                timestamp = int(key.split("_")[-1]) * 3  # Convert index to time (3s intervals)
                timestamps.append(timestamp)
                gaze_states.append(value)
                    
        # Ensure non-empty data before sorting
        if timestamps and gaze_states:
            sorted_data = sorted(zip(timestamps, gaze_states))
            timestamps, gaze_states = zip(*sorted_data)  # Unzip back into separate lists

            print("Gaze States:", gaze_states)
            print("Timesteps:", timestamps)

            plt.plot(timestamps, gaze_states, marker='o', linestyle='-', 
                     color=colors[idx % len(colors)], label=f'Participant {idx+1}')
            
            # Store data for mean calculation
            for t, g in zip(timestamps, gaze_states):
                if t not in all_gaze_data:
                    all_gaze_data[t] = []
                all_gaze_data[t].append(g)
        else:
            print(f"No valid data in {file}")

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Calculate mean gaze score at each timestep
mean_timesteps = sorted(all_gaze_data.keys())
mean_gaze_scores = [np.mean(all_gaze_data[t]) for t in mean_timesteps]

# Compute and plot mean gaze scores
if all_gaze_data:
    mean_timesteps = sorted(all_gaze_data.keys())
    mean_gaze_scores = [np.mean(all_gaze_data[t]) for t in mean_timesteps]

    plt.plot(mean_timesteps, mean_gaze_scores, linestyle='--', color='red', linewidth=2, label="Mean Gaze Score")


# Add reference lines based on condition
if condition == 'high':
    ref_times = [0, 50, 91, 180, 181, 270]
    ref_gaze_scores = [0, 0, 5, 5, 5, 5]
    plt.axhline(y=5, color='black', linestyle='--', linewidth=2, label="Reference (5)")
elif condition == 'low':
    plt.axhline(y=0, color='black', linestyle='--', linewidth=2, label="Reference (0)")
elif condition == 'dynamic':
    # Reference line pattern (0s to 90s: 0, 91s to 180s: 5, 181s to 270s: 0)
    ref_times = [0, 90, 91, 180, 181, 270]
    ref_gaze_scores = [0, 0, 5, 5, 0, 0]
    plt.plot(ref_times, ref_gaze_scores, '--', linewidth=2, color='black', label="Reference Line")

# Set axis limits to start from 0
plt.xlim(left=0)
plt.ylim(bottom=0)

# Labels, legend, and grid
plt.xlabel('Time (seconds)')
plt.ylabel('Gaze Score')
plt.title(f'Gaze Score Transition Over Time - {condition.capitalize()} Condition')
plt.legend()
plt.grid(True)

# Save plot
plt.savefig(f'gaze_plot_{condition}.png')
plt.close()
print(f"Plot saved as gaze_plot_{condition}.png")
