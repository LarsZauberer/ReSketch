from nn_test import Test_NN
from nn_train import train
from result_stats.learn_plotter import Learn_Plotter
from bayes_opt import BayesianOptimization
from agent_modules.nn_agent import DeepQNetwork, Agent
from agent_modules.environment import ShapeDraw
from data.ai_data import AI_Data
import json


current_best = None


def create_runner(args):
    def runner(gamma, epsilon, alpha, replace_target, episode_mem_size, n_episodes, friction, vel_1, vel_2):
        parameters = locals()
        global current_best

        log = logging.getLogger("optimizer-runner")
        
        # Fix parameters
        canvas_size = 28
        patch_size = 5
        n_actions = 42
        episode_mem_size = int(episode_mem_size)
        batch_size = 64
        n_episodes = int(n_episodes) + int(episode_mem_size)
        n_steps = 64
        n_epochs = 1
        
        # further calculations
        glob_in_dims = (4, canvas_size, canvas_size)
        loc_in_dims = (2, patch_size, patch_size)
        mem_size = int(episode_mem_size) * n_steps
        
        # load training data
        learn_plot = Learn_Plotter(path="src/result_stats/plotlearn_data.json")
        data = AI_Data(dataset="mnist")
        data.sample(n_episodes)
        log.debug(f"Loaded Training data")
        
        # Creating environment and agent
        env = ShapeDraw(canvas_size, patch_size, data.pro_data, n_actions=n_actions, max_action_strength=1, friction=friction, vel_1=vel_1, vel_2=vel_2)
        agent_args = {"gamma": gamma, "epsilon": epsilon, "alpha": alpha, "replace_target": replace_target, 
                    "global_input_dims": glob_in_dims, "local_input_dims": loc_in_dims, 
                    "mem_size": mem_size, "batch_size": batch_size, 
                    "q_next_dir": "src/nn_memory/q_next", "q_eval_dir": "src/nn_memory/q_eval", "n_actions": n_actions}
        agent = Agent(**agent_args)
        log.debug(f"Initiated Environmnet and Agent")
        log.info(f"Training with parameters: {parameters}")
        
        # Training
        log.debug(f"Start training")
        train(env=env,
              agent=agent,
              data=data,
              learn_plot=learn_plot,
              n_episodes=n_episodes,
              n_steps=n_steps,
              n_epochs=n_epochs,
              n_actions=n_actions,
              episode_mem_size=episode_mem_size,
              save_training=False,
              vis_compare=100,
              mnist=args.mnist,
              speed=args.speed,
              )
        log.info(f"Training finished")
        
        log.debug(f"Run Tests")
        tester = Test_NN(n_test=260, num_steps=n_steps, dataset=args.dataset)
        reward, accuracy, datarec, speed = tester.test(agent=agent, t_reward=True, t_accuracy=True, t_datarec=True, t_speed=True, t_vis=False)
        
        scores = {"reward": reward, "accuracy": accuracy, "datarec": datarec, "speed": speed}
        log.debug(f"Tests finished. Scores: {scores}")
        
        log.info(f"Testing score on criteria: {args.criteria} - {scores[args.criteria]}")
        
        # Check if should save the agent
        if current_best is None:
            save_model(args, agent, scores)
        elif scores[args.criteria] > current_best:
            save_model(args, agent, scores)
        
        return scores[args.criteria]
    
    return runner


def save_model(args, agent, scores):
    log = logging.getLogger("optimizer-save")
    
    # Same trained model weights
    log.info(f"New best score. Saving model...")
    agent.save_models()
    log.info(f"Model saved!")
    current_best = scores[args.criteria]
    log.debug(f"New best score: {current_best}")
    
    # Save hyperparameters
    log.info(f"Saving hyperparameters...")
    with open("src_physics/opti.json", "r") as f:
        data = json.load(f)
    
    if args["args"].mnist and args["args"].speed:
        log.debug(f"Saving mnist speed")
        del(args["args"])
        data["mnist_speed"] = args
    elif args["args"].mnist:
        log.debug(f"Saving mnist")
        del(args["args"])
        data["mnist"] = args
    elif args["args"].speed:
        log.debug(f"Saving speed")
        del(args["args"])
        data["speed"] = args
    else:
        log.debug(f"Saving base")
        del(args["args"])
        data["base"] = args
    
    log.debug(f"Saving data: {data}")
    with open("src_physics/opti.json", "w") as f:
        json.dump(data, f)


def main(args):
    # Hyperparameters to optimize
    bounds = {"gamma": (0.1, 1), "epsilon": (0, 1), "alpha": (
        0.00001, 0.001), "replace_target": (1000, 10000), "episode_mem_size": (100, 1000), "n_episodes": (100, 4000), "friction": (0.1, 0.7), "vel_1": (0.7, 1.2), "vel_2": (1.2, 2)}

    runner = create_runner(args)

    optimizer = BayesianOptimization(
        f=runner,
        pbounds=bounds,
        random_state=1,
    )
    
    optimizer.maximize(init_points=args.r, n_iter=args.i)
    
    log.info(f"Best found parameters: {optimizer.max}")


if __name__ == '__main__':
    import logging
    import sys
    import os
    import argparse
    from pathlib import Path
    import time
    try:
        from rich.logging import RichHandler
    except ModuleNotFoundError:
        logging.warning(
            "Rich is not installed. Please install rich by typing: pip install rich")
        sys.exit(1)

    # Initiate argparsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", help="Verbose",
                        action="store_true", default=False)
    parser.add_argument("-r", help="Number of random iterations", action="store", type=int, default=3)
    parser.add_argument("-i", help="Number of iterations", action="store", type=int, default=5)
    parser.add_argument("-d", "--dataset", help="Name of the dataset to run the test on", action="store", type=str, default="mnist")
    parser.add_argument("-c", "--criteria", help="Criteria to improve the test on", action="store", type=str, default="accuracy")
    parser.add_argument("-m", "--mnist", help="Weather to use mnist training or not", action="store_true", default=False)
    parser.add_argument("-s", "--speed", help="Weather to use speed training or not", action="store_true", default=False)
    # parser.add_argument("-t", help="Number of episodes in training", action="store", type=int, default=1000)

    args = parser.parse_args()
    
    # Check if logs folder exists
    if not Path("logs").exists():
        os.mkdir("logs")

    # File Log handler
    FileHandler = logging.FileHandler(Path("logs/log.log"))
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(name)s] [%(levelname)-5.5s] %(message)s")
    FileHandler.setFormatter(logFormatter)

    # Initiate logger
    if args.v:
        FORMAT = "%(message)s"
        logging.basicConfig(
            level="DEBUG", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(), FileHandler]
        )
    else:
        FORMAT = "%(message)s"
        logging.basicConfig(
            level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(), FileHandler]
        )

    log = logging.getLogger()
    log.debug(f"Logging initialized!")
    starttime = time.time()

    # Main Stuff
    main(args)

    log.debug(f"Application closed at {time.time() - starttime} seconds")
