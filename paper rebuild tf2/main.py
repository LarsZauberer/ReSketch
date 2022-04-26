import os
from environment import ShapeDraw
from nn_agent2 import DeepQNetwork, Agent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

if __name__ == '__main__': 
    load_checkpoint = False

    # Hyper parameters
    canvas_size = 28
    patch_size = 7
    n_actions = 2*(patch_size**2)
    mem_size = 20000
    batch_size = 64
    num_episodes = 12000
    num_steps = 50

    #loading reference Data
    ref_Data = pd.read_csv("paper rebuild tf2/ref_Data.csv")
    ref_Data = ref_Data.drop('Unnamed: 0', axis=1)
    train_ind = np.random.choice(ref_Data.shape[1], num_episodes)
    reference = []
    for i in train_ind:
        reference.append(ref_Data.iloc[i].to_numpy().reshape(28,28))

    #Initializing architecture
    env = ShapeDraw(canvas_size, patch_size, reference)
    agent = Agent(gamma=0.99, epsilon=0.5, alpha=0.001, global_input_dims=(4, canvas_size, canvas_size), 
                    local_input_dims=(2, patch_size, patch_size), mem_size=mem_size, batch_size=batch_size)
    if load_checkpoint:
        agent.load_models()
    
    #Fill replay buffer
    g_obs, l_obs = env.reset()
    for j in range(mem_size):
            action = random.randint(0, n_actions-1)
            next_g_obs, next_l_obs, reward = env.step(action)
            agent.store_transition(g_obs, l_obs, next_g_obs, next_l_obs, action, reward)
            g_obs = next_g_obs
            l_obs = next_l_obs
    
    score = 0
    scores = []
    eps_history = []
    #Main process
    for i in range(num_episodes):
        global_obs, local_obs = env.reset()
        score = 0

        for j in range(num_steps):
            action = agent.choose_action(global_obs, local_obs)
            next_gloabal_obs, next_local_obs, reward = env.step(action)

            agent.store_transition(global_obs, local_obs, next_gloabal_obs, next_local_obs, action, reward)
            global_obs = next_gloabal_obs
            local_obs = next_local_obs
            
            score += reward

            if j % 4 == 0:
                agent.learn()

        # Learn Process visualization
        if i % 12 == 0 and i > 0:
            ind = agent.mem_cntr % agent.mem_size
            print(agent.action_memory[ind-20:ind])
            avg_score = np.mean(scores[max(0, i-12):(i+1)])
            print('episode: ', i,'score: ', score,
                 ' average score %.3f' % avg_score,
                'epsilon %.3f' % agent.epsilon)
            env.render("Compare")
        else:
            print('episode: ', i,'score: ', score)
        eps_history.append(agent.epsilon)
        scores.append(score)

        if i % 100 == 0 and i > 0: 
            agent.save_models()
    
    