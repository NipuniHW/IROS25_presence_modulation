import pdb
from rewards import low_gaze_reward, high_gaze_reward, medium_gaze_reward, low_gaze_reward_LVM

#region mdp state and actions spaces

actions_incremental = {
    "Increase L, Increase M, Increase V": [1, 1, 1],
    "Increase L, Increase M, Keep V": [1, 1, 0],
    "Increase L, Increase M, Decrease V": [1, 1, -1],
    "Increase L, Keep M, Increase V": [1, 0, 1],
    "Increase L, Keep M, Keep V": [1, 0, 0],
    "Increase L, Keep M, Decrease V": [1, 0, -1],
    "Increase L, Decrease M, Increase V": [1, -1, 1],
    "Increase L, Decrease M, Keep V": [1, -1, 0],
    "Increase L, Decrease M, Decrease V": [1, -1, -1],
    "Keep L, Increase M, Increase V": [0, 1, 1],
    "Keep L, Increase M, Keep V": [0, 1, 0],
    "Keep L, Increase M, Decrease V": [0, 1, -1],
    "Keep L, Keep M, Increase V": [0, 0, 1],
    "Keep L, Keep M, Keep V": [0, 0, 0],  # No changes
    "Keep L, Keep M, Decrease V": [0, 0, -1],
    "Keep L, Decrease M, Increase V": [0, -1, 1],
    "Keep L, Decrease M, Keep V": [0, -1, 0],
    "Keep L, Decrease M, Decrease V": [0, -1, -1],
    "Decrease L, Increase M, Increase V": [-1, 1, 1],
    "Decrease L, Increase M, Keep V": [-1, 1, 0],
    "Decrease L, Increase M, Decrease V": [-1, 1, -1],
    "Decrease L, Keep M, Increase V": [-1, 0, 1],
    "Decrease L, Keep M, Keep V": [-1, 0, 0],
    "Decrease L, Keep M, Decrease V": [-1, 0, -1],
    "Decrease L, Decrease M, Increase V": [-1, -1, 1],
    "Decrease L, Decrease M, Keep V": [-1, -1, 0],
    "Decrease L, Decrease M, Decrease V": [-1, -1, -1],
}

states_gaze_score = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10
}

def generate_states_gaze_score_with_L_M_V():
    states_gaze_score_with_L_M_V = {}
    for g in range(11):  # Assuming G can be 0 or 1
        for l in range(11):  # Assuming L can be 0
            for m in range(11):  # Assuming M can be 0
                for v in range(11):  # Assuming V can be 0
                    key = f"G{g}, L{l}, M{m}, V{v}"
                    value = [g, l, m, v]
                    states_gaze_score_with_L_M_V[key] = value
    return states_gaze_score_with_L_M_V

led_actuators = [
        'ChestBoard/Led/Blue/Actuator/Value', 'ChestBoard/Led/Green/Actuator/Value', 'ChestBoard/Led/Red/Actuator/Value',
        'Ears/Led/Left/0Deg/Actuator/Value', 'Ears/Led/Left/108Deg/Actuator/Value', 'Ears/Led/Left/144Deg/Actuator/Value',
        'Ears/Led/Left/180Deg/Actuator/Value', 'Ears/Led/Left/216Deg/Actuator/Value', 'Ears/Led/Left/252Deg/Actuator/Value',
        'Ears/Led/Left/288Deg/Actuator/Value', 'Ears/Led/Left/324Deg/Actuator/Value', 'Ears/Led/Left/36Deg/Actuator/Value',
        'Ears/Led/Left/72Deg/Actuator/Value', 'Ears/Led/Right/0Deg/Actuator/Value', 'Ears/Led/Right/108Deg/Actuator/Value',
        'Ears/Led/Right/144Deg/Actuator/Value', 'Ears/Led/Right/180Deg/Actuator/Value', 'Ears/Led/Right/216Deg/Actuator/Value',
        'Ears/Led/Right/252Deg/Actuator/Value', 'Ears/Led/Right/288Deg/Actuator/Value', 'Ears/Led/Right/324Deg/Actuator/Value',
        'Ears/Led/Right/36Deg/Actuator/Value', 'Ears/Led/Right/72Deg/Actuator/Value', 'Face/Led/Blue/Left/0Deg/Actuator/Value',
        'Face/Led/Blue/Left/135Deg/Actuator/Value', 'Face/Led/Blue/Left/180Deg/Actuator/Value', 'Face/Led/Blue/Left/225Deg/Actuator/Value',
        'Face/Led/Blue/Left/270Deg/Actuator/Value', 'Face/Led/Blue/Left/315Deg/Actuator/Value', 'Face/Led/Blue/Left/45Deg/Actuator/Value',
        'Face/Led/Blue/Left/90Deg/Actuator/Value', 'Face/Led/Blue/Right/0Deg/Actuator/Value', 'Face/Led/Blue/Right/135Deg/Actuator/Value',
        'Face/Led/Blue/Right/180Deg/Actuator/Value', 'Face/Led/Blue/Right/225Deg/Actuator/Value', 'Face/Led/Blue/Right/270Deg/Actuator/Value',
        'Face/Led/Blue/Right/315Deg/Actuator/Value', 'Face/Led/Blue/Right/45Deg/Actuator/Value', 'Face/Led/Blue/Right/90Deg/Actuator/Value',
        'Face/Led/Green/Left/0Deg/Actuator/Value', 'Face/Led/Green/Left/135Deg/Actuator/Value', 'Face/Led/Green/Left/180Deg/Actuator/Value',
        'Face/Led/Green/Left/225Deg/Actuator/Value', 'Face/Led/Green/Left/270Deg/Actuator/Value', 'Face/Led/Green/Left/315Deg/Actuator/Value',
        'Face/Led/Green/Left/45Deg/Actuator/Value', 'Face/Led/Green/Left/90Deg/Actuator/Value', 'Face/Led/Green/Right/0Deg/Actuator/Value',
        'Face/Led/Green/Right/135Deg/Actuator/Value', 'Face/Led/Green/Right/180Deg/Actuator/Value', 'Face/Led/Green/Right/225Deg/Actuator/Value',
        'Face/Led/Green/Right/270Deg/Actuator/Value', 'Face/Led/Green/Right/315Deg/Actuator/Value', 'Face/Led/Green/Right/45Deg/Actuator/Value',
        'Face/Led/Green/Right/90Deg/Actuator/Value', 'Face/Led/Red/Left/0Deg/Actuator/Value', 'Face/Led/Red/Left/135Deg/Actuator/Value',
        'Face/Led/Red/Left/180Deg/Actuator/Value', 'Face/Led/Red/Left/225Deg/Actuator/Value', 'Face/Led/Red/Left/270Deg/Actuator/Value',
        'Face/Led/Red/Left/315Deg/Actuator/Value', 'Face/Led/Red/Left/45Deg/Actuator/Value', 'Face/Led/Red/Left/90Deg/Actuator/Value',
        'Face/Led/Red/Right/0Deg/Actuator/Value', 'Face/Led/Red/Right/135Deg/Actuator/Value', 'Face/Led/Red/Right/180Deg/Actuator/Value',
        'Face/Led/Red/Right/225Deg/Actuator/Value', 'Face/Led/Red/Right/270Deg/Actuator/Value', 'Face/Led/Red/Right/315Deg/Actuator/Value',
        'Face/Led/Red/Right/45Deg/Actuator/Value', 'Face/Led/Red/Right/90Deg/Actuator/Value'
    ]    
#endregion

#region Learning Configurations

class GazeFormulationBaseClass:
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)
            
        if 'states_generator' not in config_dict:
            self.states_generator = None
        else:
            self.states_generator = config_dict['states_generator']
            self.states = self.states_generator()

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = GazeFormulationBaseClass(vars(self))
        
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, GazeFormulationBaseClass):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)
    
    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)

# TODO:: Possibly change to the formulation if not universal rule
# Threshold
# Q-Learning: Low 0-3 with rounding, Medium 4-6 with rounding, High 7-10 with rounding
# Raw_score is doubles: Low 0.0-30 with rounding, Medium 31-60 with rounding, High 61-100

low_gaze_config = GazeFormulationBaseClass({
    'learning_rate': 0.1,
    'discount_factor': 0.9,
    'exploration_rate': 0.1,
    'episodes': 10000,
    'epsilon': 0.9,
    'epislon_decay': 0.99,
    'gamma': 0.9,
    'reward_function': low_gaze_reward,
    'actions': actions_incremental,
    'states': states_gaze_score,
    'gaze_threshold': [0, 3],
    'led_actuators': led_actuators
})

low_gaze_config_with_L_M_V = GazeFormulationBaseClass({
    'states_generator': generate_states_gaze_score_with_L_M_V,
    'learning_rate': 0.1,
    'discount_factor': 0.9,
    'exploration_rate': 0.1,
    'episodes': 10000,
    'epsilon': 0.9,
    'epislon_decay': 0.99,
    'gamma': 0.9,
    'reward_function': low_gaze_reward_LVM,
    'actions': actions_incremental,
    'states': states_gaze_score,
    'gaze_threshold': [0, 3]
})

medium_gaze_config = GazeFormulationBaseClass({
    'learning_rate': 0.1,
    'discount_factor': 0.9,
    'exploration_rate': 0.1,
    'episodes': 10000,
    'epsilon': 0.9,
    'epislon_decay': 0.99,
    'gamma': 0.9,
    'reward_function': medium_gaze_reward,
    'actions': actions_incremental,
    'states': states_gaze_score,
    'gaze_threshold': [4, 6]
})

high_gaze_config = GazeFormulationBaseClass({
    'learning_rate': 0.1,
    'discount_factor': 0.9,
    'exploration_rate': 0.1,
    'episodes': 10000,
    'epsilon': 0.9,
    'epislon_decay': 0.99,
    'gamma': 0.9,
    'reward_function': high_gaze_reward,
    'actions': actions_incremental,
    'states': states_gaze_score,
    'gaze_threshold': [5, 6]
})

#endregion