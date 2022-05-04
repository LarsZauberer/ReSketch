import os
from agent_modules.environment import ShapeDraw
from agent_modules.nn_agent2 import DeepQNetwork, Agent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import keyboard
import json


if __name__ == '__main__': 
    #memory parameters
    load_checkpoint = False
    isSaved = False
    lastsave = 0

    # Hyper parameters
    canvas_size = 28
    patch_size = 7
    n_actions = 2*(patch_size**2)
    mem_size = 20000
    batch_size = 64
    num_episodes = 12000
    num_steps = 64

    #loading reference Data
    ref_Data = pd.read_csv("paper rebuild stable/ref_Data.csv")
    ref_Data = ref_Data.drop('Unnamed: 0', axis=1)
    train_ind = np.random.choice(ref_Data.shape[1], num_episodes+100)
    reference = []
    for i in train_ind:
        reference.append(ref_Data.iloc[i].to_numpy().reshape(28,28))

    #Initializing architecture
    env = ShapeDraw(canvas_size, patch_size, reference)
    agent = Agent(gamma=0.99, epsilon=0, alpha=0.001, global_input_dims=(4, canvas_size, canvas_size), 
                    local_input_dims=(2, patch_size, patch_size), mem_size=mem_size, batch_size=batch_size, replace_target=1000)
    if load_checkpoint:
        agent.load_models()
    
    #Fill replay buffer
    g_obs, l_obs = env.reset()
    for j in range(mem_size):
            action = random.randint(0, n_actions-1)
            next_g_obs, next_l_obs, reward = env.step(action)
            agent.store_transition(g_obs, l_obs, next_g_obs, next_l_obs, action, reward)
            agent.counter += 1
            g_obs = next_g_obs
            l_obs = next_l_obs
    
    score = 0
    scores = []
    learning_history = [[],[]]
    #Main process
    for i in range(num_episodes):
        global_obs, local_obs = env.reset()
        score = 0

        for j in range(num_steps):
            action = agent.choose_action(global_obs, local_obs)
            next_gloabal_obs, next_local_obs, reward = env.step(action)
            #env.render("Compare", realtime=True)
            
            agent.store_transition(global_obs, local_obs, next_gloabal_obs, next_local_obs, action, reward)
            global_obs = next_gloabal_obs
            local_obs = next_local_obs
            
            agent.counter += 1
            score += reward

            if (j+1) % 4 == 0:
                agent.learn()

        # Learn Process visualization
        if i % 12 == 0 and i > 0:
            ind = agent.counter % agent.mem_size
            print(agent.action_memory[ind-20:ind])
            avg_score = np.mean(scores[max(0, i-12):(i+1)])
            print('episode: ', i,'score: ', score, ' average score %.3f' % avg_score, 'epsilon %.3f' % agent.epsilon)
            env.render("Compare")
            learning_history[0].append(i)
            learning_history[1].append(avg_score)
        else:
            print('episode: ', i,'score: ', score)
        scores.append(score)

        #bad memory fix: save manually
        if not isSaved:
            if keyboard.is_pressed("s"):
                lastsave = i
                isSaved = True
                agent.save_models()
                with open("paper rebuild stable/plotlearn_data.json", "w") as f:
                    json.dump(learning_history, f)
        elif i > lastsave+2:
            isSaved = False

        

    agent.save_models()
    
    