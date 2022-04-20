import os
from environment import ShapeDraw
from nn_agent2 import DeepQNetwork, Agent
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


if __name__ == '__main__': 

    

    ref_Data = pd.read_csv("C:/Users/robin/OneDrive/Desktop/Maturarbeit/Nachzeichner-KI/paper rebuild/ref_Data.csv")
    
    ref_Data = ref_Data.drop('Unnamed: 0', axis=1)
    reference = []
    for i in range(ref_Data.shape[1]):
        reference.append(ref_Data.iloc[i].to_numpy().reshape(28,28))




    env = ShapeDraw(28, 28, reference)
    load_checkpoint = False
    agent = Agent(gamma=0.99, epsilon=0, alpha=0.000025, global_input_dims=(4,28,28), local_input_dims=(2,7,7),
                n_actions=7*7*2, mem_size=25000, batch_size=64)
    if load_checkpoint:
        agent.load_models()
    scores = []
    eps_history = []
    num_episodes = 50000
    num_steps = 150
    
    score = 0
    
    #ToDo: Fill memory?

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
            avg_score = np.mean(scores[max(0, i-12):(i+1)])
            print('episode: ', i,'score: ', score,
                 ' average score %.3f' % avg_score,
                'epsilon %.3f' % agent.epsilon)
            agent.save_models()
        else:
            print('episode: ', i,'score: ', score)
        eps_history.append(agent.epsilon)
        scores.append(score)
    
    