import csv
import yaml

def csv_to_yaml(csv_file, yaml_file):
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)  # Read CSV as dictionary
            data = [row for row in reader]  # Convert rows into a list of dictionaries

        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"Conversion successful! YAML saved as {yaml_file}")

    except Exception as e:
        print(f"Error: {e}")

# Example usage
csv_to_yaml('/home/nipuni/Documents/IROS25_presence_modulation/collected data/Random/P8 Dynamic.csv', '/home/nipuni/Documents/IROS25_presence_modulation/collected data/Random/P8 Dynamic.yaml')
