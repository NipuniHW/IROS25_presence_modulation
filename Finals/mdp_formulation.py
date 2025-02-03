from rewards import low_gaze_reward, high_gaze_reward, medium_gaze_reward

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
    'gaze_threshold': [0.0, 30.0]
})

#endregion


