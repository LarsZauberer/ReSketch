from agent_modules.environment import ShapeDraw
from agent_modules.nn_agent import DeepQNetwork, Agent
#from data.ai_data import AI_Data
from models.mnist_model.models import EfficientCapsNet




from time import sleep

from rich.progress import track
import numpy as np

import onnx
from onnx_tf.backend import prepare
import json

from keras.models import model_from_json

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import cm, colors


class Test_NN():
    def __init__(self, n_test: int = 260, num_steps: int = 64, image_num=10, version="base", dataset : str = "mnist"):
        self.n_test = n_test
        self.num_steps = num_steps
        self.version = version
        self.dataset = dataset

        # Load Hyperparameter data
        with open("src/opti.json", "r") as f:
            hyp_data = json.load(f)
            
        if version == "mnist-speed":
            hyp_data = hyp_data["mnist-speed"]
        elif version == "mnist":
            hyp_data = hyp_data["mnist"]
        elif version == "speed":
            hyp_data = hyp_data["speed"]
        else:
            hyp_data = hyp_data["base"]
        
        print(f"Hyperparameters: {hyp_data}")

        # initialize environment
        canvas_size = 28
        patch_size = 7
        self.n_actions = 2*patch_size**2
        self.dataset = dataset
        self.data = AI_Data(dataset)
        self.data.sample(n_test)
        self.envir = ShapeDraw(canvas_size, patch_size, self.data.pro_data)

        # initialize agent
        self.agent_args = {"gamma": hyp_data["gamma"], "epsilon": 0, "alpha": hyp_data["alpha"], "replace_target": int(hyp_data["replace_target"]), 
                            "global_input_dims": (4, canvas_size, canvas_size) , "local_input_dims": (2, patch_size, patch_size), 
                            "mem_size": int(int(hyp_data["episode_mem_size"])*num_steps), "batch_size": 64, "model": f"base-{self.version}"}

        # images
        self.image_indexes = iter(sorted(np.random.choice(n_test, image_num, replace=False)) + [0])
        self.curInd = next(self.image_indexes)
        self.images = []

        # for speed test
        self.done_accuracy = 0.75
        
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
        ep_counter = 0

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
                self.envir.curStep = j
                # Run the timestep
                action = agent.choose_action(global_obs, local_obs, illegal_list=illegal_moves)
                # TODO: Parametrize the min_decrement and the rec_reward value
                next_gloabal_obs, next_local_obs, reward = self.envir.step(action, decrementor=1, rec_reward=0.1, without_rec=True, min_decrement=0.3)
                

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
            if t_vis:
                    if ep_counter == self.curInd:
                        self.images.append(self.envir.gradient_render())
                        self.curInd = next(self.image_indexes)
            
            ep_counter += 1
                    

        scores = []
        if t_reward: scores.append(np.mean(reward_scores))
        if t_accuracy: scores.append(np.mean(accuracy_scores))
        if t_datarec: scores.append(np.mean(datarec_scores))
        if t_speed: scores.append(np.mean(speed_scores))
        if t_vis: self.generate_image(columns=2)
                
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


    def generate_image(self, columns=2):
        num = len(self.images)
        rows = int(num/columns)
        
        labeled = 0

        fig = plt.figure(figsize=(10., 15.))
        grid = ImageGrid(fig, 111,  
                        nrows_ncols=(rows, columns*3), 
                        axes_pad=0.1,  
                        )

        interval = np.full((28, 1), 255)
        sorted_images = []
        for index, item in enumerate(self.images):
            ref, canv = item
            sorted_images.append((ref.reshape((28,28)), False))
            sorted_images.append((canv.reshape((28,28)), True))
            sorted_images.append(interval.copy())

        for ax, im in zip(grid, sorted_images):
            # Iterating over the grid returns the Axes.
            ax.axis("off")
            if len(im) == 2:
                ax.imshow(im[0], cmap="bone", vmin=0, vmax=255)
                if labeled < columns*2:
                    if im[1]:
                        ax.set_title("ReSketch")
                        labeled += 1
                    else:
                        if labeled == 0:
                            ax.set_title("MNIST")
                        elif labeled == 2:
                            ax.set_title("EMNIST")
                        else:
                            ax.set_title("QuickDraw")
                        labeled += 1
            else:
                ax.imshow(im, cmap="bone", vmin=0, vmax=255)
        
        fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=64), cmap="bone"), ax=grid, orientation="horizontal", fraction=0.046, pad=0.04, label="Steps", location="bottom")

        plt.savefig(f"src/images/base-{self.version}-{self.dataset}.png", bbox_inches='tight')
        plt.pause(5)
        
        
    
    

        

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
            score = self.test(agent, t_reward=True, t_accuracy=True, t_datarec=True, t_speed=True, t_vis=True)

        score = [float('%.3f' % s) for s in score]
        return score



if __name__ == '__main__':  
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", help="Numer of test episodes", action="store", type=int, default=100)
    parser.add_argument("-d", "--dataset", help="Name of the dataset to run the test on", action="store", type=str, default="mnist")
    parser.add_argument("-c", "--criterion", help="The criterion to test on", action="store", type=str, default="all")
    parser.add_argument("-v", "--version", help="The version to test", action="store", type=str, default="base")
    parser.add_argument("-s", "--save", help="Save Results", action="store_true", default=False)
    parser.add_argument("--image", help="Generate Image of all datasets", action="store_true", default=False)
    args = parser.parse_args()
    
    if args.image:
        images = []
        test = Test_NN(n_test=200, dataset="mnist", version=args.version)
        test.test_from_loaded(agent_args=test.agent_args, mode="vis")
        images1 = test.images[:8]
        
        test = Test_NN(n_test=200, dataset="emnist", version=args.version)
        test.test_from_loaded(agent_args=test.agent_args, mode="vis")
        images2 = test.images[:8]
        
        test = Test_NN(n_test=200, dataset="quickdraw", version=args.version)
        test.test_from_loaded(agent_args=test.agent_args, mode="vis")
        images3 = test.images[:8]
        
        for i in range(8):
            images.append(images1[i])
            images.append(images2[i])
            images.append(images3[i])
        
        test.images = images
        test.generate_image(columns=3)
        exit()

    test = Test_NN(n_test=args.test, dataset=args.dataset, version=args.version)
    reward, accuracy, datarec, speed = test.test_from_loaded(test.agent_args, mode=args.criterion)
    
    if args.save:
        with open(Path(f"results/base-{args.version}-{args.dataset}-{args.criterion}.txt"), "w") as f:
            f.write(f'reward: {reward}, accuracy: {accuracy}, {test.dataset} recognition: {datarec}, speed {speed}')
    
    print(f'reward: {reward}, accuracy: {accuracy}, {test.dataset} recognition: {datarec}, speed {speed}')

