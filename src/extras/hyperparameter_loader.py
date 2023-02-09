from pathlib import Path
import json
import logging

from extras.logger import critical

@critical
def hyperparameter_loader(path: str, modelName: str):
    """
    hyperparameter_loader Load the hyperparameters from the file

    :param path: the path to the hyperparameter file
    :type path: str
    :param modelName: The name of the model
    :type modelName: str
    :return: Dictionary of the different hyperparameters of the model
    :rtype: dict
    """
    # Load Hyperparameter data
    with open(Path(path), "r") as f:
        hyp_data = json.load(f)
    if hyp_data.get(modelName, False):
        hyp_data = hyp_data[modelName]
    else:
        log = logging.getLogger("hyperparameter_loader")
        log.warning(f"This is not a known model -> Returning default hyperparameters")
        #No known Model: give default Values
        hyp_data = {"gamma": 0.70, "epsilon": 0.3, "epsilon_episodes": 1000, "alpha": 0.0002, "replace_target": 6000, "episode_mem_size": 50, "n_episodes": 3000, "friction": 0.3, "vel_1": 0.9, "vel_2": 1.5} 
    
    return hyp_data