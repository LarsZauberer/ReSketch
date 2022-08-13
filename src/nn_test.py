from agent_modules.environment import ShapeDraw
from agent_modules.nn_agent import DeepQNetwork, Agent
from data.ai_data import AI_Data
from mnist_model.models import EfficientCapsNet

from rich.progress import track
import numpy as np



class Test_NN():
    def __init__(self, n_test: int = 100, num_steps: int = 64):
        self.n_test = n_test
        self.num_steps = num_steps

        canvas_size = 28
        patch_size = 7
        self.glob_in_dims = (5, canvas_size, canvas_size)
        self.loc_in_dims = (2, patch_size, patch_size)
        self.n_actions = 2*patch_size**2
        self.episode_mem_size = 700
        self.batch_size = 64
        

        self.done_accuracy = 0.6

        self.test_data = []
        self.sorted_data = [] 
        self.data = AI_Data(path="src/data/test_ref_Data.json")
        self.data.sample(n_test)

        self.envir = ShapeDraw(canvas_size, patch_size, self.data.pro_data)

        #for mnist test
        self.mnist_model = EfficientCapsNet('MNIST', mode='test', verbose=False)
        self.mnist_model.load_graph_weights()


    def test(self, agent: Agent, t_reward: bool = False, t_accuracy: bool = False, t_mnist : bool = False, t_speed : bool = False):
        """ 
        Tests a given Model for n_test Episodes
        
        :param agent: the model to be tested
        :type agent: Agent
        :param t_reward: Test according to accumulated reward
        :type t_reward: bool
        :param t_accuracy: Test according to percentual accuracy
        :type t_accuracy: bool
        :param t_mnist: Test according to percentual correct mnist recognition
        :type t_mnist: bool
        :param t_speed: Test according to steps until finished
        :type t_speed: bool
        
        :return: the Average Accuracy of each criterion of each Episode in the test
        :rtype: list
        """
        self.data.shuffle()
        self.envir.referenceData = self.data.pro_data

        reward_scores = []
        accuracy_scores = []
        mnist_scores = []
        speed_scores = []

        for i in track(range(self.n_test), description="testing"):
            global_obs, local_obs = self.envir.reset()
            score = 0
            done_step = 64

            for j in range(self.num_steps):
                illegal_moves = np.zeros(self.n_actions)
                illegal_moves = self.envir.illegal_actions(illegal_moves)
                # Run the timestep
                action = agent.choose_action(global_obs, local_obs, illegal_list=illegal_moves)
                next_gloabal_obs, next_local_obs, reward = self.envir.step(action)

                global_obs = next_gloabal_obs
                local_obs = next_local_obs
                agent.counter += 1

                if t_reward:
                    score += reward
                if t_speed:
                    if self.envir.agent_is_done(self.done_accuracy): 
                        done_step = j
                        break

            if t_reward: 
                reward_scores.append(score)
            if t_accuracy: 
                accuracy_scores.append(1 - self.envir.lastSim)
            if t_mnist:
                ref, canv = self.envir.predict_mnist()
                mnist_scores.append(int(ref == canv))
            if t_speed:
                speed_scores.append(done_step)

        scores = []
        if t_reward: scores.append(np.mean(reward_scores))
        if t_accuracy: scores.append(np.mean(accuracy_scores))
        if t_mnist: scores.append(np.mean(mnist_scores))
        if t_speed: scores.append(np.mean(speed_scores))
                
           
        return scores

    
    def test_from_loaded(self, agent_args : dict, mode : str = "all"):
        """ 
        Test Model from saved weights
            
        :param agent_args: the parameters of the model to be tested
        :type agent_args: dict
        :param mode: the mode of testing (possibilities: 'reward', 'accuracy', 'mnist', 'speed')
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
        elif mode == "mnist":
            score = self.test(agent, t_mnist=True)
        elif mode == "speed":
            score = self.test(agent, t_speed=True)
        else:
            score = self.test(agent, t_reward=True, t_accuracy=True, t_mnist=True, t_speed=True)

        score = [float('%.3f' % s) for s in score]
        return score





if __name__ == '__main__':  
    test = Test_NN()
    agent_args = {"gamma": 0.66, "epsilon": 0, "alpha": 0.00075, "replace_target": 8000, 
                  "global_input_dims": test.glob_in_dims , "local_input_dims": test.loc_in_dims, 
                  "mem_size": test.episode_mem_size*test.num_steps, "batch_size": test.batch_size, 
                  "q_next_dir": "src/nn_memory/q_next", "q_eval_dir": "src/nn_memory/q_eval"}

    reward, accuracy, mnist, speed = test.test_from_loaded(agent_args)
    print(f'reward: {reward}, accuracy: {accuracy}, mnist: {mnist}, speed {speed}')
