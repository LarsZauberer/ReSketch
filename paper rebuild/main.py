import os
from environment import ShapeDraw
from nn_agent import DeepQNetwork, Agent
import numpy as np

import matplotlib.pyplot as plt


if __name__ == '__main__':

    """
    ToDo:
        - load training data
        - convert to bitmap?
        - shuffle training data
        - 
    """ 

    env = ShapeDraw(84, 84, [5,5])
    load_checkpoint = False
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.000025, input_dims=(180,160,4),
                n_actions=3, mem_size=25000, batch_size=64)
    if load_checkpoint:
        agent.load_models()
    filename = 'breakout-alpha0p000025-gamma0p9-only-one-fc-2.png'
    scores = []
    eps_history = []
    num_episodes = 50000
    num_steps = 300
    
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
    
    