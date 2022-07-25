from agent_modules.environment import ShapeDraw
from agent_modules.nn_agent import DeepQNetwork, Agent
from data.ai_data import AI_Data
from mnist_model.models import EfficientCapsNet

from rich.progress import track
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import keyboard
import json
import time


class Test_NN():
    def __init__(self, n_test: int = 100, num_steps: int = 64):
        self.n_test = n_test
        self.num_steps = num_steps

        canvas_size = 28
        patch_size = 7
        self.n_actions = 2*patch_size**2
        self.test_data = []
        self.sorted_data = [] 
        
        self.data = AI_Data(path="src/data/test_ref_Data.json")
        self.data.sample(n_test)

        self.envir = ShapeDraw(canvas_size, patch_size, self.data.pro_data)

        #for mnist test
        self.mnist_model = EfficientCapsNet('MNIST', mode='test', verbose=False)
        self.mnist_model.load_graph_weights()

    def test(self, agent: Agent):
        """ 
        Tests a given Model for n_test Episodes
        
        :param agent: the model to be tested
        :type agent: Agent
        :return: the Average Accuracy of each Episode in the test
        :rtype: float
        """

        print("...Testing...")
        self.data.shuffle()
        self.envir.referenceData = self.data.pro_data
        scores = []
        for i in range(self.n_test):
            global_obs, local_obs = self.envir.reset()
            score = 0

            for j in range(self.num_steps):
                illegal_moves = np.zeros(self.n_actions)
                illegal_moves = self.envir.illegal_actions(illegal_moves)
                # Run the timestep
                action = agent.choose_action(global_obs, local_obs, illegal_list=illegal_moves)
                next_gloabal_obs, next_local_obs, reward = self.envir.step(action)

                global_obs = next_gloabal_obs
                local_obs = next_local_obs

                agent.counter += 1
                score += reward

            if i % 12 == 0 and i > 0:
                self.envir.render("Compare")
            
            scores.append(score)
           
        avg_score = np.mean(scores)

        return avg_score
    

    def test_from_loaded(self, agent_args: dict):
        """ 
        Test Model from saved weights
            
        :param agent_args: the parameters of the model to be tested
        :type agent_args: dict
        :return: the Average Accuracy of each Episode in the test
        :rtype: float
        """

        # Initializing architecture
        agent = Agent(**agent_args)
        agent.load_models()

        score = self.test(agent)
        return '%.3f' % score


    
    def mnist_test(self, agent: Agent):
         
        print("...Testing...")
        self.data.shuffle()
        self.envir.referenceData = self.data.pro_data
        
        scores = 0
        for i in range(self.n_test):
            global_obs, local_obs = self.envir.reset()
            
            for j in range(self.num_steps):
                # Run the timestep
                action = agent.choose_action(global_obs, local_obs)
                next_gloabal_obs, next_local_obs, reward = self.envir.step(action, i*self.num_steps+j)

                global_obs = next_gloabal_obs
                local_obs = next_local_obs

                agent.counter += 1
            

            

            inp = np.array([self.envir.reference, self.envir.canvas])
            out = self.mnist_model.predict(inp)
            ref = np.argmax(out[0][0])
            canv = np.argmax(out[0][1])

            scores += int(ref == canv) 

        return scores/self.n_test
        

    def mnist_test_from_loaded(self, agent_args : dict):
        """ 
        Test Model according to mnist accuracy from saved weights
            
        :param agent_args: the parameters of the model to be tested
        :type agent_args: dict
        :return: the Average Accuracy of each Episode in the test
        :rtype: float
        """

        # Initializing architecture
        agent = Agent(**agent_args)
        agent.load_models()

        score = self.mnist_test(agent)
        return '%.4f' % score






if __name__ == '__main__':  
    # Hyper parameters
    canvas_size = 28
    patch_size = 7
    episode_mem_size = 200
    batch_size = 64
    num_steps = 64
    # further calculations
    glob_in_dims = (4, canvas_size, canvas_size)
    loc_in_dims = (2, patch_size, patch_size)
    mem_size = episode_mem_size*num_steps

    kwargs = {"gamma": 0.99, "epsilon": 0, "alpha": 0.005, "replace_target": 1000, 
            "global_input_dims": glob_in_dims, "local_input_dims": loc_in_dims, 
            "mem_size": mem_size, "batch_size": batch_size, 
             "q_next_dir": "src/nn_memory/q_next", "q_eval_dir": "src/nn_memory/q_eval"}

    test = Test_NN()
    print(f'score: {test.mnist_test_from_loaded(kwargs)}')
