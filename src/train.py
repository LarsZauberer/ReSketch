import argparse
import json

from data_statistics.learn_plotter import Learn_Plotter
from data.ai_data import AI_Data
from reproduce_modules.environment import Environment as Rep_Env
from reproduce_modules.nn_agent import Agent as Rep_Agent

from physics_modules.environment import Environment as Phy_Env
from physics_modules.nn_agent import Agent as Phy_Agent

from train_functions import hyperparameter_loader, train



def reproduce():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mnist", help="Whether to use mnist training or not", action="store_true", default=False)
    parser.add_argument("-s", "--speed", help="Whether to use speed training or not", action="store_true", default=False)
    parser.add_argument("-n", "--modelName", help="Name of Model to be trained", action="store", default="new_model")
    parser.add_argument("-d", "--dataset", help="Name of Dataset to train with", action="store", default="mnist_train")
    args = parser.parse_args()

    model_name= args.modelName
    model_path = f"pretrained_models/reproduce/{model_name}"

    # Options
    mnist = args.mnist
    speed = args.speed

    hyp_data = hyperparameter_loader("src/opti.json", model_name)

    #Manual hyperparameters:
    #hyp_data = {"gamma": 0.7, "epsilon": 0, "alpha": 0.0002, "replace_target": 6000, "episode_mem_size": 900, "n_episodes": 3000} 

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


    env = Rep_Env(canvas_size, patch_size, data.pro_data)
    agent_args = {"gamma": hyp_data["gamma"], "epsilon": hyp_data["epsilon"], "alpha": hyp_data["alpha"], "replace_target": int(hyp_data["replace_target"]), 
                  "global_input_dims": glob_in_dims, "local_input_dims": loc_in_dims, 
                  "mem_size": mem_size, "batch_size": batch_size}
    agent = Rep_Agent(**agent_args)
    

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
        mnist=mnist,
        speed=speed
        )



def physics():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mnist", help="Whether to use mnist training or not", action="store_true", default=False)
    parser.add_argument("-s", "--speed", help="Whether to use speed training or not", action="store_true", default=False)
    parser.add_argument("-n", "--modelName", help="Name of Model to be trained", action="store", default="new_model")
    parser.add_argument("-d", "--dataset", help="Name of Dataset to train with", action="store", default="mnist_train")
    args = parser.parse_args()

    model_name= args.modelName
    model_path = f"pretrained_models/physics/{model_name}"

    # Options
    mnist = args.mnist
    speed = args.speed

    hyp_data = hyperparameter_loader("src/phy_opti.json", model_name)

    #Manual hyperparameters:
    #hyp_data = {"gamma": 0.7, "epsilon": 0, "alpha": 0.0002, "replace_target": 6000, "episode_mem_size": 900, "n_episodes": 3000} 

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


    env =  Phy_Env(canvas_size, patch_size, data.pro_data, n_actions=n_actions, friction=hyp_data["friction"], vel_1=hyp_data["vel_1"], vel_2=hyp_data["vel_2"])
    agent_args = {"gamma": hyp_data["gamma"], "epsilon": hyp_data["epsilon"], "alpha": hyp_data["alpha"], "replace_target": int(hyp_data["replace_target"]), 
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
        vis_compare=-12,
        mnist=mnist,
        speed=speed
        )






if __name__ == "__main__":
    #reproduce()

    physics()


