import argparse
from pathlib import Path
import logging

from data.ai_data import AI_Data
from reproduce_modules.environment import Environment as Rep_Env
from reproduce_modules.nn_agent import Agent as Rep_Agent

from physics_modules.environment import Environment as Phy_Env
from physics_modules.nn_agent import Agent as Phy_Agent

    
from extras.test_functions import test_env
from extras.hyperparameter_loader import hyperparameter_loader

from extras.logger import critical, initialize_logging
from data_statistics.Image_Generator import generate_image



@critical
def physics_test(args):
    """
    physics_test Tests a physics model

    :param args: Arguments from argparsing
    :type args: Namespace
    """
    log = logging.getLogger("Tester")
    
    data = AI_Data(args.dataset)
    data.sample(args.test)
    # initialize environment
    canvas_size = 28
    patch_size = 5
    n_actions = 42
    glob_in_dims = (4, canvas_size, canvas_size)
    loc_in_dims = (2, patch_size, patch_size)
    hyp_data = hyperparameter_loader("src/phy_opti.json", args.version.split("-")[1])
    
    env =  Phy_Env(canvas_size, patch_size, data.pro_data, n_actions=n_actions, friction=hyp_data["friction"], vel_1=hyp_data["vel_1"], vel_2=hyp_data["vel_2"])
    agent_args = {"gamma": 0, "epsilon": 0, "alpha": 0, "replace_target": 1000, 
                  "global_input_dims": glob_in_dims, "local_input_dims": loc_in_dims, 
                  "mem_size": 1000, "batch_size": 64, "n_actions": n_actions}
    agent = Phy_Agent(**agent_args)
    agent.load_models(f"pretrained_models/reproduce/{args.version}")

    scores = test_env(
        env=env,
        agent=agent,
        data=data,
        n_episodes=args.test,
        )
    
    reward, accuracy, datarec, speed = [float('%.3f' % s) for s in scores[-1]]
    log.info(f'reward: {reward}, accuracy: {accuracy}, {data.dataset}-recognition: {datarec}, speed {speed}')
    generate_image(scores[-1])

    if args.save:
        with open(Path(f"results/reproduce-{args.version}-{args.dataset}.txt"), "w") as f:
            f.write(f'reward: {reward}, accuracy: {accuracy}, {data.dataset} recognition: {datarec}, speed {speed}')




@critical
def reproduce_test(args):
    """
    reproduce_test Testing a reproduce model

    :param args: Arguments from argparsing
    :type args: Namespace
    """
    log = logging.getLogger("Tester")
    
    data = AI_Data(args.dataset)
    data.sample(args.test)

    # initialize environment
    canvas_size = 28
    patch_size = 7
    env = Rep_Env(canvas_size, patch_size, data.pro_data)

    env = Rep_Env(canvas_size, patch_size, data.pro_data)
    agent_args = {"gamma": 0, "epsilon": 0, "alpha": 0, "replace_target": 1000, 
                  "global_input_dims": (4, canvas_size, canvas_size), "local_input_dims": (2, patch_size, patch_size), 
                  "mem_size": 1000, "batch_size": 64, "epsilon_episodes": 0}
    agent = Rep_Agent(**agent_args)
    agent.load_models(f"pretrained_models/reproduce/{args.version}")

    scores = test_env(
        env=env,
        agent=agent,
        data=data,
        n_episodes=args.test
        )

    
    reward, accuracy, datarec, speed = [float('%.3f' % s) for s in scores[:-1]]
    log.info(f'reward: {reward}, accuracy: {accuracy}, {data.dataset}-recognition: {datarec}, speed {speed}')
    generate_image(scores[-1])

    if args.save:
        with open(Path(f"results/reproduce-{args.version}-{args.dataset}.txt"), "w") as f:
            f.write(f'reward: {reward}, accuracy: {accuracy}, {data.dataset} recognition: {datarec}, speed {speed}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", help="Number of test episodes", action="store", type=int, default=100)
    parser.add_argument("-d", "--dataset", help="Name of the dataset to run the test on", action="store", type=str, default="mnist_test")
    parser.add_argument("-c", "--criterion", help="The criterion to test on", action="store", type=str, default="all")
    parser.add_argument("-v", "--version", help="The version to test", action="store", type=str, default="base-base")
    parser.add_argument("-s", "--save", help="Save Results", action="store_true", default=False)
    parser.add_argument("--image", help="Generate Image of all datasets", action="store_true", default=False)
    parser.add_argument("--debug", help="Verbose for the logging", action="store_true", default=False)
    args = parser.parse_args()
    
    initialize_logging(args)
    log = logging.getLogger()
    
    # Print all the configuration
    log.debug(f"Number of test episodes: {args.test}")
    log.debug(f"Dataset: {args.dataset}")
    log.debug(f"Criterion to test for: {args.criterion}")
    log.debug(f"Model Name: {args.version}")
    log.debug(f"Should the results be saved? - {args.save}")
    
    if "physics" in args.version:
        log.info(f"Recognized a physics model")
        physics_test(args)
    else:
        log.info(f"Recognized a reproduce model")
        reproduce_test(args)












    

    
