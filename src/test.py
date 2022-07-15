from agent_modules.environment import ShapeDraw
from agent_modules.nn_agent2 import DeepQNetwork, Agent
from data.data_prep import sample_data, shuffle_data

from rich.progress import track
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import keyboard
import json


if __name__ == '__main__':


    n_test = 1000
    test_data = []

    # Hyper parameters
    canvas_size = 28
    patch_size = 7
    n_actions = 2*(patch_size**2)
    episode_mem_size = 200
    batch_size = 64
    num_episodes = 4000
    num_steps = 64
    epochs = 3

    # further calculations
    glob_in_dims = (4, canvas_size, canvas_size)
    loc_in_dims = (2, patch_size, patch_size)
    mem_size = episode_mem_size*num_steps
  

    #loading data
    sorted_data = [] 
    with open("src/data/test_ref_Data.json", "r") as f:
        sorted_data = json.load(f)
    test_data = sample_data(sorted_data, n_test)
    test_data = shuffle_data(test_data)

    # Initializing architecture
    env = ShapeDraw(canvas_size, patch_size, test_data)
    agent = Agent(gamma=0.99, epsilon=0, alpha=0.001, replace_target=1000,
                  global_input_dims=glob_in_dims, local_input_dims=loc_in_dims, 
                  mem_size=mem_size, batch_size=batch_size, 
                  q_next_dir='src/nn_memory/q_next', q_eval_dir='src/nn_memory/q_eval')
    agent.load_models()

    scores = []
    for i in range(n_test):
        global_obs, local_obs = env.reset()
        score = 0

        for j in range(num_steps):
            # Run the timestep
            action = agent.choose_action(global_obs, local_obs)
            next_gloabal_obs, next_local_obs, reward = env.step(action)
            #env.render("Compare", realtime=True)

            agent.counter += 1
            score += reward
        
        scores.append(score)
        print(f"score: {score} ")


    print("\n########################################\n")
    avg_score = np.mean(scores)
    print(f"average score: {avg_score}")
    

            











    