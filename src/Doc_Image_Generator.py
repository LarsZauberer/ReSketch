import argparse
from pathlib import Path
import logging

from data.ai_data import AI_Data
from modules.environment import Environment as Rep_Env
from modules.nn_agent import Agent as Rep_Agent

from extras.physics_modules.environment import Environment as Phy_Env
from extras.physics_modules.nn_agent import Agent as Phy_Agent

import numpy as np
    
from test_functions import test_env, generative_test_env

from extras.logger import critical, initialize_logging
from data_statistics.Image_Generator import generate_image, generate_generative_image


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import cm, colors



@critical
def reproduce_generate(model_name="0.1", testnum = 100, rows = 8):
    # load data
    data = AI_Data("mnist_test")
    data.sample(testnum)
    # initialize environment
    canvas_size = 28
    patch_size = 7
    env = Rep_Env(canvas_size, patch_size, data.labeled_pro_data, with_stopAction=True)
    agent_args = {"softmax": False, "gamma": 0, "epsilon_episodes": 1000, "epsilon": 0, "alpha": 0, "replace_target": 1000, 
                  "global_input_dims": (4, canvas_size, canvas_size), "local_input_dims": (2, patch_size, patch_size), 
                  "mem_size": 1000, "batch_size": 64}
    agent = Rep_Agent(**agent_args)
    agent.load_models(f"pretrained_models/reproduce/{model_name}")
    # Run Test
    scores = test_env(
        env=env,
        agent=agent,
        data=data,
        n_episodes=testnum
        )

    mnist_images = scores.pop(-1)[:rows]

    print(scores)

    # load data
    data = AI_Data("emnist_test")
    data.sample(testnum)
    env.referenceData = data.labeled_pro_data

    # Run Test
    scores = test_env(
        env=env,
        agent=agent,
        data=data,
        n_episodes=testnum
        )

    emnist_images = scores.pop(-1)[:rows]

    data = AI_Data("quickdraw_test")
    data.sample(testnum)
    env.referenceData = data.labeled_pro_data

    # Run Test
    scores = test_env(
        env=env,
        agent=agent,
        data=data,
        n_episodes=testnum
        )
    
    quickdraw_images = scores.pop(-1)[:rows]

    fig = plt.figure(figsize=(10., 15.))
    grid = ImageGrid(fig, 111,  
                    nrows_ncols=(rows, 8), 
                    axes_pad=0.1,  
                    )

    interval = np.full((28, 1), 255)

    sorted_images = []
    for im_mnist, im_emnist, im_quickdraw in zip(mnist_images, emnist_images, quickdraw_images):
        
        sorted_images.append(im_mnist[0].reshape((28,28)))
        sorted_images.append(im_mnist[1].reshape((28,28)))
        
        sorted_images.append(interval.copy())

        sorted_images.append(im_emnist[0].reshape((28,28)))
        sorted_images.append(im_emnist[1].reshape((28,28)))

        sorted_images.append(interval.copy())

        sorted_images.append(im_quickdraw[0].reshape((28,28)))
        sorted_images.append(im_quickdraw[1].reshape((28,28)))


    labeled = 0
    titles = iter(["Referenz", "MNIST", " ", "Referenz", "EMNIST", " ", "Referenz", "QuickDraw", " ", " "])
    for ax, im in zip(grid, sorted_images):
        # Iterating over the grid returns the Axes.
        ax.axis("off")
        if labeled < 8:
            ax.set_title(next(titles))
            labeled += 1
    
        ax.imshow(im, cmap="bone", vmin=0, vmax=255)
    

    fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=64), cmap="bone"), ax=grid, orientation="horizontal", fraction=0.046, pad=0.01, label="Steps", location="bottom")

    plt.savefig(f"src/data_statistics/ai_images.png", bbox_inches='tight')
    plt.pause(10)


if __name__ == "__main__":
    reproduce_generate()








@critical
def generative_generate(model_name="g0.1", testnum = 26, softmax=False, noisy=False, rows = 7):
    # initialize environment
    canvas_size = 28
    patch_size = 7
    env = Rep_Env(canvas_size, patch_size, referenceData=np.full((testnum, 28, 28), 1), generative=True, with_stopAction=True, with_noisy=noisy)

    agent_args = {"softmax": softmax, "gamma": 0, "epsilon_episodes": 1000, "epsilon": 0, "alpha": 0, "replace_target": 1000, 
                  "global_input_dims": (4, canvas_size, canvas_size), "local_input_dims": (2, patch_size, patch_size), 
                  "mem_size": 1000, "batch_size": 64}
    agent = Rep_Agent(**agent_args)
    agent.load_models(f"pretrained_models/generative/{model_name}-0")
    agent.set_softmax_temp(0.08)
 
    # Run Test
    scores = generative_test_env(
        env=env,
        agent=agent,
        n_episodes=testnum
        )
    
    zero_images = scores.pop(-1)

    agent_args = {"softmax": softmax, "gamma": 0, "epsilon_episodes": 1000, "epsilon": 0, "alpha": 0, "replace_target": 1000, 
                  "global_input_dims": (4, canvas_size, canvas_size), "local_input_dims": (2, patch_size, patch_size), 
                  "mem_size": 1000, "batch_size": 64}
    agent = Rep_Agent(**agent_args)
    agent.load_models(f"pretrained_models/generative/{model_name}-2")
    agent.set_softmax_temp(0.08)
 
    # Run Test
    scores = generative_test_env(
        env=env,
        agent=agent,
        n_episodes=testnum
        )
    
    two_images = scores.pop(-1)

    agent_args = {"softmax": softmax, "gamma": 0, "epsilon_episodes": 1000, "epsilon": 0, "alpha": 0, "replace_target": 1000, 
                  "global_input_dims": (4, canvas_size, canvas_size), "local_input_dims": (2, patch_size, patch_size), 
                  "mem_size": 1000, "batch_size": 64}
    agent = Rep_Agent(**agent_args)
    agent.load_models(f"pretrained_models/generative/{model_name}-2")
    agent.set_softmax_temp(0.08)
 
    # Run Test
    scores = generative_test_env(
        env=env,
        agent=agent,
        n_episodes=testnum
        )
    
    eight_images = scores.pop(-1)

    agent_args = {"softmax": softmax, "gamma": 0, "epsilon_episodes": 1000, "epsilon": 0, "alpha": 0, "replace_target": 1000, 
                  "global_input_dims": (4, canvas_size, canvas_size), "local_input_dims": (2, patch_size, patch_size), 
                  "mem_size": 1000, "batch_size": 64}
    agent = Rep_Agent(**agent_args)
    agent.load_models(f"pretrained_models/generative/{model_name}-2")
    agent.set_softmax_temp(0.08)
 
    # Run Test
    scores = generative_test_env(
        env=env,
        agent=agent,
        n_episodes=testnum
        )
    
    f_images = scores.pop(-1)


    agent_args = {"softmax": softmax, "gamma": 0, "epsilon_episodes": 1000, "epsilon": 0, "alpha": 0, "replace_target": 1000, 
                  "global_input_dims": (4, canvas_size, canvas_size), "local_input_dims": (2, patch_size, patch_size), 
                  "mem_size": 1000, "batch_size": 64}
    agent = Rep_Agent(**agent_args)
    agent.load_models(f"pretrained_models/generative/{model_name}-2")
    agent.set_softmax_temp(0.08)
 
    # Run Test
    scores = generative_test_env(
        env=env,
        agent=agent,
        n_episodes=testnum
        )
    
    flower_images = scores.pop(-1)



    fig = plt.figure(figsize=(10., 15.))
    grid = ImageGrid(fig, 111,  
                    nrows_ncols=(rows, 5), 
                    axes_pad=0.1,  
                    )

    interval = np.full((28, 1), 255)

    sorted_images = []
    for im_zero, im_two, im_eight, im_f, im_flower in zip(zero_images, two_images, eight_images, f_images, flower_images):
        
        sorted_images.append(im_zero.reshape((28,28)))
        sorted_images.append(im_two.reshape((28,28)))
        sorted_images.append(im_eight.reshape((28,28)))
        sorted_images.append(im_f.reshape((28,28)))
        sorted_images.append(im_flower.reshape((28,28)))


    labeled = 0
    titles = iter(["Null", "Zwei", "Acht", "F", "Blume"])
    for ax, im in zip(grid, sorted_images):
        # Iterating over the grid returns the Axes.
        ax.axis("off")
        if labeled < 5:
            ax.set_title(next(titles))
            labeled += 1
    
        ax.imshow(im, cmap="bone", vmin=0, vmax=255)
    

    fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=64), cmap="bone"), ax=grid, orientation="horizontal", fraction=0.046, pad=0.01, label="Steps", location="bottom")

    plt.savefig(f"src/data_statistics/ai_images.png", bbox_inches='tight')
    plt.pause(10)

   


