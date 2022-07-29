from agent_modules.environment import ShapeDraw
from agent_modules.nn_agent import DeepQNetwork, Agent
from data.ai_data import AI_Data

from rich.progress import track
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import keyboard
import json
import time

from rich.progress import track


class Test_NN():
    def __init__(self, n_test: int = 10, num_steps: int = 64):
        self.n_test = n_test
        self.num_steps = num_steps

        canvas_size = 28
        patch_size = 5
        self.n_actions = 42
        self.test_data = []
        self.sorted_data = [] 
        
        self.data = AI_Data(path="src/data/test_ref_Data.json")
        self.data.sample(n_test)

        self.envir = ShapeDraw(canvas_size, patch_size, self.data.pro_data, n_actions=self.n_actions, do_render=False)

    def test(self, agent: Agent):
        print("...Testing...")
        self.data.shuffle()
        self.envir.referenceData = self.data.pro_data
        
        scores = []
        for i in track(range(self.n_test), description="testing"):
            global_obs, local_obs = self.envir.reset()
            score = 0

            for j in range(self.num_steps):
                illegal_moves = np.zeros(self.n_actions)
                illegal_moves = self.envir.illegal_actions(illegal_moves)
                # Run the timestep
                if not all(a == 1 for a in illegal_moves):
                    action = agent.choose_action(global_obs, local_obs, illegal_moves, )
                else:
                    action = np.random.choice(n_actions)
                
                next_gloabal_obs, next_local_obs, reward = self.envir.step(action)

                global_obs = next_gloabal_obs
                local_obs = next_local_obs

                agent.counter += 1
                score += reward
            
            scores.append(score)
           
        avg_score = np.mean(scores)

        return avg_score
    

    def test_from_loaded(self, agent_args):
        # Initializing architecture
        agent = Agent(**agent_args)
        agent.load_models()

        score = self.test(agent)
        return '%.3f' % score
       



if __name__ == '__main__':  

     # Hyper parameters
    canvas_size = 28
    patch_size = 5
    episode_mem_size = 603
    batch_size = 64
    n_steps = 64
    n_epochs = 3
    max_action_strength = 1
    n_actions = 42

    # further calculations
    glob_in_dims = (4, canvas_size, canvas_size)
    loc_in_dims = (2, patch_size, patch_size)
    mem_size = episode_mem_size*n_steps

   
    kwargs = {"gamma": 0.7156814785141222, "epsilon": 0.25, "alpha": 0.0003739100350232336, "n_actions" : n_actions, "replace_target": 4000, 
                  "global_input_dims": glob_in_dims, "local_input_dims": loc_in_dims, 
                  "mem_size": mem_size, "batch_size": batch_size, 
                  "q_next_dir": "src/nn_memory/q_next", "q_eval_dir": "src/nn_memory/q_eval"}


    test = Test_NN()
    print(f'score: {test.test_from_loaded(kwargs)}')



    









    