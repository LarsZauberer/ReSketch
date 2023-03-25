import argparse
import logging

from data_statistics.learn_plotter import Learn_Plotter
from data.ai_data import AI_Data
from modules.environment import Environment as Rep_Env
from modules.nn_agent import Agent as Rep_Agent

from extras.physics_modules.environment import Environment as Phy_Env
from extras.physics_modules.nn_agent import Agent as Phy_Agent

from train_functions import train

from extras.logger import initialize_logging, critical


@critical
def physics(args):
    """
    physics Train a physics model

    :param args: Arguments from argparsing
    :type args: Namespace
    """

    # Model name + path
    model_name= args.modelName
    model_path = f"pretrained_models/reproduce/{model_name}"

    # Hyperparameters
    hyp_data = {"gamma": 0.70, "epsilon": 0.3, "epsilon_episodes": 1000, "alpha": 0.0002, "replace_target": 6000, "episode_mem_size": 900, "n_episodes": 5000, "friction": 0.3, "vel_1": 0.9, "vel_2": 1.5} 

    # Agent, Environment constants
    canvas_size = 28
    patch_size = 5
    n_actions = 42
    episode_mem_size = int(hyp_data["episode_mem_size"])
    batch_size = 64
    n_episodes = int(hyp_data["n_episodes"]) + episode_mem_size
    n_steps = 64
    # further calculations
    glob_in_dims = (4, canvas_size, canvas_size)
    loc_in_dims = (2, patch_size, patch_size)
    mem_size = episode_mem_size*n_steps

    # load Data
    learn_plot = Learn_Plotter(path="src/data_statistics/plotlearn_data.json")
    data = AI_Data(dataset=args.dataset)
    data.sample(n_episodes)

    # create Environment, Agent
    env =  Phy_Env(canvas_size, patch_size, data.labeled_pro_data, n_actions=n_actions, friction=hyp_data["friction"], vel_1=hyp_data["vel_1"], vel_2=hyp_data["vel_2"])
    agent_args = {"softmax": args.softmax, "gamma": hyp_data["gamma"], "epsilon": hyp_data["epsilon"], "epsilon_episodes": hyp_data["epsilon_episodes"], "alpha": hyp_data["alpha"], "replace_target": int(hyp_data["replace_target"]), 
                  "global_input_dims": glob_in_dims, "local_input_dims": loc_in_dims, 
                  "mem_size": mem_size, "batch_size": batch_size, "n_actions": n_actions}
    agent = Phy_Agent(**agent_args)
    
    # Start training
    train(
        env=env,
        agent=agent,
        data=data,
        learn_plot=learn_plot,
        episode_mem_size=episode_mem_size,
        n_episodes=n_episodes,
        n_steps=n_steps,
        model_path=model_path,
        save_training=True,
        vis_compare=-12
        )


@critical
def reproduce(args):
    """
    reproduce Train a reproducee model

    :param args: Argparsing arguments
    :type args: Namespace
    """

    # Model name + path
    model_name= args.modelName
    model_path = f"pretrained_models/reproduce/{model_name}"

    # Hyperparameters
    hyp_data = {"gamma": 0.7, "epsilon_episodes": 2000, "epsilon": 0.3, "alpha": 0.00015, "replace_target": 6000, "episode_mem_size": 900, "n_episodes": 5000} 

    # Agent, Environment constants
    canvas_size = 28
    patch_size = 7
    episode_mem_size = int(hyp_data["episode_mem_size"])
    batch_size = 64
    n_episodes = int(hyp_data["n_episodes"]) + episode_mem_size
    n_steps = 64
    # further calculations
    glob_in_dims = (4, canvas_size, canvas_size)
    loc_in_dims = (2, patch_size, patch_size)
    mem_size = episode_mem_size*n_steps

    # load Data
    learn_plot = Learn_Plotter(path="src/data_statistics/plotlearn_data.json")
    data = AI_Data(dataset=args.dataset)
    data.sample(n_episodes)

    # create Environment, Agent
    env = Rep_Env(canvas_size, patch_size, data.labeled_pro_data, with_stopAction=args.stopAction, with_liftpen=args.liftpen, with_overdraw=args.overdraw, with_noisy=False, generative=False, dataset=args.dataset.split("_")[0])
    agent_args = {"softmax": args.softmax, "gamma": hyp_data["gamma"], "epsilon": hyp_data["epsilon"], "epsilon_episodes": hyp_data["epsilon_episodes"], "alpha": hyp_data["alpha"], "replace_target": int(hyp_data["replace_target"]), 
                  "global_input_dims": glob_in_dims, "local_input_dims": loc_in_dims, 
                  "mem_size": mem_size, "batch_size": batch_size}
    agent = Rep_Agent(**agent_args)
    agent.set_softmax_temp(0.05)
    
    # Start training
    train(
        env=env,
        agent=agent,
        data=data,
        learn_plot=learn_plot,
        episode_mem_size=episode_mem_size, 
        n_episodes=n_episodes,
        n_steps=n_steps,
        model_path=model_path,
        save_training=True,
        vis_compare=-12,
        )


def generative(args):
    """
    generative Train a generative model

    :param args: Argparsing arguments
    :type args: Namespace
    """

    # model name + path
    model_name= args.modelName
    model_path = f"pretrained_models/generative/{model_name}"

    # Hyperparameters
    hyp_data = {"gamma": 0.8, "epsilon_episodes": 2000, "epsilon": 0.3, "alpha": 0.0002, "replace_target": 6000, "episode_mem_size": 900, "n_episodes": 6000} 

    # Agent, Environment constants
    canvas_size = 28
    patch_size = 7
    episode_mem_size = int(hyp_data["episode_mem_size"])
    batch_size = 64
    n_episodes = int(hyp_data["n_episodes"]) + episode_mem_size
    n_steps = 64
    # further calculations
    glob_in_dims = (4, canvas_size, canvas_size)
    loc_in_dims = (2, patch_size, patch_size)
    mem_size = episode_mem_size*n_steps

    # load Data
    learn_plot = Learn_Plotter(path="src/data_statistics/plotlearn_data.json")
    data = AI_Data(dataset=args.dataset, motive=int(args.genMotive))
    data.sample_by_category(data.motive, n_episodes)

    # create Environment + Agent
    env = Rep_Env(canvas_size, patch_size, data.labeled_pro_data, with_stopAction=2, with_liftpen=args.liftpen, with_overdraw=args.overdraw, with_noisy=args.noisyPixel, generative=True, dataset=args.dataset.split("_")[0])
    agent_args = {"softmax": args.softmax, "gamma": hyp_data["gamma"], "epsilon": hyp_data["epsilon"], "epsilon_episodes": hyp_data["epsilon_episodes"], "alpha": hyp_data["alpha"], "replace_target": int(hyp_data["replace_target"]), 
                  "global_input_dims": glob_in_dims, "local_input_dims": loc_in_dims, 
                  "mem_size": mem_size, "batch_size": batch_size}
    agent = Rep_Agent(**agent_args)
    agent.set_softmax_temp(0.07)

    # Start training
    train(
        env=env,
        agent=agent,
        data=data,
        learn_plot=learn_plot,
        episode_mem_size=episode_mem_size,
        n_episodes=n_episodes,
        n_steps=n_steps,
        model_path=model_path,
        save_training=True,
        vis_compare=-12,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--modelName", help="Name of Model to be trained", action="store", default="new_model")
    parser.add_argument("-d", "--dataset", help="Name of Dataset to train with", action="store", default="mnist_train")

    parser.add_argument("-p", "--physics", help="Run the physics version", action="store_true", default=False)    
    parser.add_argument("-s", "--stopAction", help="Run the stopAction version", action="store", default=0)
    parser.add_argument("-o", "--overdraw", help="Run the Overdraw reward function", action="store_true", default=False)
    parser.add_argument("-l", "--liftpen", help="Run the liftpen reward function", action="store_true", default=False)

    parser.add_argument("-g", "--generative", help="Run the Generative version", action="store_true", default=False)
    parser.add_argument("-np", "--noisyPixel", help="Run the noisypixel variation", action="store_true", default=False)
    parser.add_argument("-sm", "--softmax", help="Run the NN with Softmax activation", action="store_true", default=False)
    parser.add_argument("-gm", "--genMotive", help="generative Motive", action="store", default=2)

    parser.add_argument("--debug", help="Verbose for the logging", action="store_true", default=False)
    args = parser.parse_args()
    
    initialize_logging(args)
    log = logging.getLogger()
    
    log.info(f"Dataset: {args.dataset}")
    log.info(f"Model name: {args.modelName}")
    log.info(f"Softmax Training: {args.softmax}")
    log.info(f"PHYSICS Variation: {args.physics}")
    log.info(f"GENERATIVE Variation: {args.generative}")
    log.info(f"StopAction Variation: {args.stopAction}")
    log.info(f"Overdraw Variation: {args.overdraw}")
    log.info(f"Liftpen Variation: {args.liftpen}")
    log.info(f"noisy-pixel Variation: {args.noisyPixel}")

    # start Training
    if args.physics:
        physics(args)
    elif args.generative:
        generative(args)
    else:
        reproduce(args)
