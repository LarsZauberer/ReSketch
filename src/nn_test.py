from agent_modules.environment import ShapeDraw
from agent_modules.nn_agent import DeepQNetwork, Agent
from data.ai_data import AI_Data
from models.mnist_model.models import EfficientCapsNet
from time import sleep

from rich.progress import track
import numpy as np

import onnx
from onnx_tf.backend import prepare
import json

from keras.models import model_from_json


class Test_NN():
    def __init__(self, n_test: int = 260, num_steps: int = 64, dataset : str = "mnist"):
        self.n_test = n_test
        self.num_steps = num_steps

        canvas_size = 28
        patch_size = 5
        self.glob_in_dims = (4, canvas_size, canvas_size)
        self.loc_in_dims = (2, patch_size, patch_size)
        self.n_actions = 42
        self.episode_mem_size = 700
        self.batch_size = 64
        

        self.done_accuracy = 0.6

        self.dataset = dataset
        self.data = AI_Data(dataset)
        self.data.sample(n_test)

        self.envir = ShapeDraw(canvas_size, patch_size, self.data.pro_data, n_actions=self.n_actions)

        #for mnist test
        self.mnist_model = EfficientCapsNet('MNIST', mode='test', verbose=False)
        self.mnist_model.load_graph_weights()
        
        # For quickdraw test
        # reference: https://analyticsindiamag.com/converting-a-model-from-pytorch-to-tensorflow-guide-to-onnx/
        q_model = onnx.load("src/models/quickdraw_model/quickdraw.onnx")
        self.tf_q_model = prepare(q_model)
        
        # For emnist test
        json_file = open('src/models/emnist_model/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.emnist_model = model_from_json(loaded_model_json)
        self.emnist_model.load_weights('src/models/emnist_model/model.h5')


    def test(self, agent: Agent, t_reward: bool = False, t_accuracy: bool = False, t_datarec : bool = False, t_speed : bool = False, t_vis: bool = False):
        """ 
        Tests a given Model for n_test Episodes
        
        :param agent: the model to be tested
        :type agent: Agent
        :param t_reward: Test according to accumulated reward
        :type t_reward: bool
        :param t_accuracy: Test according to percentual accuracy
        :type t_accuracy: bool
        :param t_datarec: Test according to percentual correct recognition of the drawn motive
        :type t_datarec: bool
        :param t_speed: Test according to steps until finished
        :type t_speed: bool
        
        :return: the Average Accuracy of each criterion of each Episode in the test
        :rtype: list
        """
        self.data.shuffle()
        self.envir.referenceData = self.data.pro_data

        reward_scores = []
        accuracy_scores = []
        datarec_scores = []
        speed_scores = []

        for i in track(range(self.n_test), description="testing"):
            global_obs, local_obs = self.envir.reset()
            score = 0
            done_step = 64

            for j in range(self.num_steps):
                # Run the timestep
                illegal_moves = np.zeros(self.n_actions)
                illegal_moves = self.envir.illegal_actions(illegal_moves)

                

                if not all(a == 1 for a in illegal_moves):
                    action = agent.choose_action(global_obs, local_obs, illegal_moves, replay_fill=False)
                else:
                    action = np.random.choice(self.n_actions)
                    

                next_gloabal_obs, next_local_obs, reward = self.envir.step(action)
                global_obs = next_gloabal_obs
                local_obs = next_local_obs

                agent.counter += 1

                if t_reward:
                    score += reward
                if t_speed:
                    if (1 - self.envir.lastSim) > self.done_accuracy:
                        if self.dataset == "emnist":
                            ref, canv = self.predict_emnist()
                        elif self.dataset == "quickdraw":
                            ref, canv = self.predict_quickdraw()
                        else:
                            ref, canv = self.envir.predict_mnist()
                        if ref == canv: 
                            if done_step == 64:
                                done_step = j

            if t_reward: 
                reward_scores.append(score)
            if t_accuracy: 
                accuracy_scores.append(1 - self.envir.lastSim)
            if t_datarec:
                if self.dataset == "emnist":
                    ref, canv = self.predict_emnist()
                elif self.dataset == "quickdraw":
                    ref, canv = self.predict_quickdraw()
                else:
                    ref, canv = self.envir.predict_mnist()
                datarec_scores.append(int(ref == canv))
            if t_speed:
                speed_scores.append(done_step)

        scores = []
        if t_reward: scores.append(np.mean(reward_scores))
        if t_accuracy: scores.append(np.mean(accuracy_scores))
        if t_datarec: scores.append(np.mean(datarec_scores))
        if t_speed: scores.append(np.mean(speed_scores))
                
        return scores

    def predict_quickdraw(self):
        #Reshape data
        canvas = self.envir.canvas.reshape(28 * 28)
        reference = self.envir.reference.reshape(28 * 28)
        # predict
        canv = self.tf_q_model.run(np.array([canvas], dtype=np.float32))
        ref = self.tf_q_model.run(np.array([reference], dtype=np.float32))
        return (np.argmax(ref[0]), np.argmax(canv[0]))

    def predict_emnist(self):
        # Reshape data
        canvas = self.envir.canvas.reshape(28 * 28)
        reference = self.envir.reference.reshape(28 * 28)
        # predict
        canv = self.emnist_model(np.array([canvas], dtype=np.float32))
        ref = self.emnist_model(np.array([reference], dtype=np.float32))
        return (np.argmax(ref[0]), np.argmax(canv[0]))
        

    def test_from_loaded(self, agent_args : dict, mode : str = "all"):
        """ 
        Test Model from saved weights
            
        :param agent_args: the parameters of the model to be tested
        :type agent_args: dict
        :param mode: the mode of testing (possibilities: 'reward', 'accuracy', 'datarec', 'speed')
        :type mode: str
        :return: the Average Accuracy of each Episode in the test
        :rtype: float
        """

        # Initializing architecture
        agent = Agent(**agent_args)
        agent.load_models()

        if mode == "reward":
            score = self.test(agent, t_reward=True)
        elif mode == "accuracy":
            score = self.test(agent, t_accuracy=True)
        elif mode == "datarec":
            score = self.test(agent, t_datarec=True)
        elif mode == "speed":
            score = self.test(agent, t_speed=True)
        elif mode == "vis":
            score = self.test(agent, t_vis=True)
        else:
            score = self.test(agent, t_reward=True, t_accuracy=True, t_datarec=True, t_speed=True)

        score = [float('%.3f' % s) for s in score]
        return score



if __name__ == '__main__':  
    test = Test_NN(dataset="mnist")
    agent_args = {"gamma": 0.66, "epsilon": 0, "alpha": 0.00075, "replace_target": 8000, 
                  "global_input_dims": test.glob_in_dims , "local_input_dims": test.loc_in_dims, 
                  "mem_size": test.episode_mem_size*test.num_steps, "batch_size": test.batch_size, 
                  "q_next_dir": "src/nn_memory/q_next", "q_eval_dir": "src/nn_memory/q_eval", "n_actions": test.n_actions}

    reward, accuracy, datarec, speed = test.test_from_loaded(agent_args, mode="all")
    print(f'reward: {reward}, accuracy: {accuracy}, {test.dataset} recognition: {datarec}, speed {speed}')

