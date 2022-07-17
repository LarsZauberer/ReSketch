from xmlrpc.client import FastMarshaller
from agent_modules.environment import ShapeDraw
from agent_modules.nn_agent import DeepQNetwork, Agent
from data.data_prep import sample_data, shuffle_data

from rich.progress import track
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import keyboard
import json
from time import sleep
import sys
import os


if __name__ == '__main__':
    # memory parameters
    load_checkpoint = False
    isSaved = False
    lastsave = 0

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

    agent_args = {"gamma": 0.99, "epsilon": 0, "alpha": 0.001, "replace_target": 1000, 
                  "global_input_dims": glob_in_dims, "local_input_dims": loc_in_dims, 
                  "mem_size": mem_size, "batch_size": batch_size, 
                  "q_next_dir": "src/nn_memory/q_next", "q_eval_dir": "src/nn_memory/q_eval"}
    

    #loading data
    sorted_data = [] 
    with open("src/data/train_ref_Data.json", "r") as f:
        sorted_data = json.load(f)
    reference = sample_data(sorted_data, num_episodes)

    # Initializing architecture
    env = ShapeDraw(canvas_size, patch_size, reference)
    agent = Agent(**agent_args)
    if load_checkpoint:
        agent.load_models()
    

    total_counter = 0
    learning_history = [[], []]
    scores = []
    for n in range(epochs):

        reference = shuffle_data(reference.copy())
        env.referenceData = reference
        
        if n == 0:
            run_episodes = num_episodes-episode_mem_size

            # Fill replay buffer
            # Fill the replay buffer to have something to learn in the first episode
            for i in track(range(episode_mem_size), description="Filling Replay Buffer"):
                g_obs, l_obs = env.reset()
                
                for j in range(num_steps):
                    action = random.randint(0, n_actions-1)

                    next_g_obs, next_l_obs, reward = env.step(action)
                    agent.store_transition(g_obs, l_obs, next_g_obs,
                                        next_l_obs, action, reward)
                    agent.counter += 1
                    g_obs = next_g_obs
                    l_obs = next_l_obs
        else:
            run_episodes = num_episodes


        # Main process
        for i in range(run_episodes):
            global_obs, local_obs = env.reset()
            score = 0
            total_counter += 1

            for j in range(num_steps):
                # Run the timestep
                action = agent.choose_action(global_obs, local_obs)
                
                next_gloabal_obs, next_local_obs, reward = env.step(action)
                #env.render("Compare", realtime=True)

                # Save new information
                agent.store_transition(
                    global_obs, local_obs, next_gloabal_obs, next_local_obs, action, reward)
                global_obs = next_gloabal_obs
                local_obs = next_local_obs

                agent.counter += 1
                score += reward

                if (j+1) % 4 == 0:
                    agent.learn()
            


            # Learn Process visualization
            rel_acc = score/env.maxScore
            if total_counter % 12 == 0 and total_counter > 0:
                avg_score = np.mean(scores)
                scores = []
                print(f"episode: {total_counter}, score: {score}, percent: {rel_acc}, average score: {'%.3f' % avg_score}, epsilon: {'%.3f' % agent.epsilon}")

                #env.render("Compare")
                learning_history[0].append(total_counter)
                learning_history[1].append(avg_score)
            else:
                print(f"episode: {total_counter}, score: {score}, percent {rel_acc}")

            scores.append(rel_acc)


            # bad memory fix: save manually
            # Save the learn data manually by pressing 's'
            if not isSaved:
                if keyboard.is_pressed("s"):
                    lastsave = i
                    isSaved = True
                    agent.save_models()
                    with open("src/result_stats/plotlearn_data.json", "w") as f:
                        json.dump(learning_history, f)
            elif i > lastsave+2:
                isSaved = False


agent.save_models()
with open("src/result_stats/plotlearn_data.json", "w") as f:
    json.dump(learning_history, f)



