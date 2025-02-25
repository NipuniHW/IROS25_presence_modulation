import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import yaml
import glob

# Get YAML files for Dynamic and Random conditions
dynamic_files = glob.glob("/home/nipuni/Documents/IROS25_presence_modulation/day2_exp_data/Dynamic/*.yaml")
random_files = glob.glob("/home/nipuni/Documents/IROS25_presence_modulation/collected data/Random/*.yaml")

if not dynamic_files or not random_files:
    print("No YAML files found. Check the path!")

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
all_dynamic_data = {}
all_random_data = {}

plt.figure(figsize=(10, 6))

# Process Dynamic Condition
for idx, file in enumerate(dynamic_files):
    print(f"Processing (Dynamic): {file}")
    
    try:
        with open(file, 'r') as f:
            data = yaml.safe_load(f)

        timestamps = []
        gaze_states = []
        
        for key, value in data.items():
            if key.startswith("state_subject_1_timestep_"):
                timestamp = int(key.split("_")[-1]) * 3  # Convert index to time (3s intervals)
                timestamps.append(timestamp)
                gaze_states.append(value)
                    
        # Sort timestamps and corresponding gaze scores
        sorted_data = sorted(zip(timestamps, gaze_states))
        timestamps, gaze_states = zip(*sorted_data)  # Unzip back into separate lists

        print("Dynamic Gaze States:", gaze_states)
        print("Dynamic Timesteps:", timestamps)

        if gaze_states:
            # plt.plot(timestamps, gaze_states, marker='o', linestyle='-', color=colors[idx % len(colors)], label=f'Dynamic P{idx+1}')
            
            # Store data for mean calculation
            for t, g in zip(timestamps, gaze_states):
                if t not in all_dynamic_data:
                    all_dynamic_data[t] = []
                all_dynamic_data[t].append(g)

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Process Random Condition
for idx, file in enumerate(random_files):
    print(f"Processing (Random): {file}")
    
    try:
        with open(file, 'r') as f:
            data = yaml.safe_load(f)

        timestamps = []
        gaze_states = []
        
        for entry in data:  # Assuming YAML is a list of dictionaries
            if "Index" in entry and "Gaze State" in entry:
                timestamp = (int(entry["Index"]) - 1) * 3  # Normalize "Index" (1-based) to 0-based
                timestamps.append(timestamp)
                gaze_states.append(entry["Gaze State"])
                    
        # Sort timestamps and corresponding gaze scores
        sorted_data = sorted(zip(timestamps, gaze_states))
        timestamps, gaze_states = zip(*sorted_data)

        print("Random Gaze States:", gaze_states)
        print("Random Timesteps:", timestamps)

        if gaze_states:
            # plt.plot(timestamps, gaze_states, marker='s', linestyle='--', color=colors[(idx+3) % len(colors)], label=f'Random P{idx+1}')
            
            # Store data for mean calculation
            for t, g in zip(timestamps, gaze_states):
                if t not in all_random_data:
                    all_random_data[t] = []
                all_random_data[t].append(g)

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Calculate Mean Gaze Scores for Dynamic and Random Conditions
dynamic_mean_timesteps = sorted(all_dynamic_data.keys())
random_mean_timesteps = sorted(all_random_data.keys())

# Convert gaze scores to float before calculating mean
dynamic_mean_gaze_scores = [np.mean([float(g) for g in all_dynamic_data[t]]) for t in dynamic_mean_timesteps]
random_mean_gaze_scores = [np.mean([float(g) for g in all_random_data[t]]) for t in random_mean_timesteps]


# Plot Mean Gaze Scores
plt.plot(dynamic_mean_timesteps, dynamic_mean_gaze_scores, 'k-', color = 'blue',linewidth=2, label="Mean Dynamic")
plt.plot(random_mean_timesteps, random_mean_gaze_scores, 'k--', linewidth=2, label="Mean Random")

# Add Reference Line for Dynamic Condition
ref_times = [0, 90, 91, 180, 181, 270]
ref_gaze_scores = [0, 0, 5, 5, 0, 0]
plt.plot(ref_times, ref_gaze_scores, 'r--', linewidth=2, label="Reference Line")

# Set axis limits
plt.xlim(left=0)
plt.ylim(bottom=0)

# Labels, legend, and grid
plt.xlabel('Time (seconds)')
plt.ylabel('Gaze Score')
plt.title('Gaze Score Comparison: Dynamic vs Random')
plt.legend()
plt.grid(True)

# Save plot
plt.savefig('gaze_plot_comparison.png')
plt.close()
print("Plot saved as gaze_plot_comparison.png")
