import argparse
from pathlib import Path

from data.ai_data import AI_Data
from reproduce_modules.environment import Environment as Rep_Env
from reproduce_modules.nn_agent import Agent as Rep_Agent

from physics_modules.environment import Environment as Phy_Env
from physics_modules.nn_agent import Agent as Phy_Agent

    
from test_functions import test_env








def reproduce_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", help="Numer of test episodes", action="store", type=int, default=100)
    parser.add_argument("-d", "--dataset", help="Name of the dataset to run the test on", action="store", type=str, default="mnist_test")
    parser.add_argument("-c", "--criterion", help="The criterion to test on", action="store", type=str, default="all")
    parser.add_argument("-v", "--version", help="The version to test", action="store", type=str, default="base-base")
    parser.add_argument("-s", "--save", help="Save Results", action="store_true", default=False)
    parser.add_argument("--image", help="Generate Image of all datasets", action="store_true", default=False)
    args = parser.parse_args()

    data = AI_Data(args.dataset)
    data.sample(args.test)

    # initialize environment
    canvas_size = 28
    patch_size = 7
    env = Rep_Env(canvas_size, patch_size, data.pro_data)


    env = Rep_Env(canvas_size, patch_size, data.pro_data)
    agent_args = {"gamma": 0, "epsilon": 0, "alpha": 0, "replace_target": 1000, 
                  "global_input_dims": (4, canvas_size, canvas_size), "local_input_dims": (2, patch_size, patch_size), 
                  "mem_size": 1000, "batch_size": 64}
    agent = Rep_Agent(**agent_args)
    agent.load_models(f"pretrained_models/reproduce/{args.version}")


    scores = test_env(
        env=env,
        agent=agent,
        data=data,
        n_episodes=args.test,
        t_reward=True,
        t_accuracy=True,
        t_datarec=True,
        t_speed=True,
        t_vis=False
        )
    
    reward, accuracy, datarec, speed = [float('%.3f' % s) for s in scores]

    print(f'reward: {reward}, accuracy: {accuracy}, {data.dataset}-recognition: {datarec}, speed {speed}')

    
    
    if args.save:
        with open(Path(f"results/reproduce-{args.version}-{args.dataset}.txt"), "w") as f:
            f.write(f'reward: {reward}, accuracy: {accuracy}, {data.dataset} recognition: {datarec}, speed {speed}')
    
    

    
if __name__ == "__main__":
    reproduce_test()



















    
        
        
    
    

        

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
    
    

    
