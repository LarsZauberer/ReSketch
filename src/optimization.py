from dataclasses import replace
from agent_modules.environment import ShapeDraw
from agent_modules.nn_agent import DeepQNetwork, Agent
from data.ai_data import AI_Data
from result_stats.learn_plotter import Learn_Plotter
import numpy as np
import json
from bayes_opt import BayesianOptimization
from rich.progress import track




# Hyper parameters
canvas_size = 28
patch_size = 7
n_actions = 2*(patch_size**2)
batch_size = 64
n_episodes = 1000
n_steps = 64
n_epochs = 1

# further calculations
glob_in_dims = (4, canvas_size, canvas_size)
loc_in_dims = (2, patch_size, patch_size)


#load Data
learn_plot = Learn_Plotter(path="src/result_stats/plotlearn_data.json")
data = AI_Data(path="src/data/train_ref_Data.json")
data.sample(n_episodes)
data.shuffle()




def runner(gamma):

    epsilon = 0.2
    alpha = 0.0005
    replace_target = 4000 
    episode_mem_size = 900

    mem_size = int(episode_mem_size*n_steps)
    replace_target = int(replace_target)
    



    rec_reward = 0.2
    rec_cons = 8
    rec_thres = 0.8
    


    env = ShapeDraw(canvas_size, patch_size, data.pro_data)
    agent_args = {"gamma": gamma, "epsilon": epsilon, "alpha": alpha, "replace_target": replace_target, 
                "global_input_dims": glob_in_dims, "local_input_dims": loc_in_dims, 
                "mem_size": mem_size, "batch_size": batch_size, 
                "q_next_dir": "src/nn_memory/q_next", "q_eval_dir": "src/nn_memory/q_eval"}
    agent = Agent(**agent_args)
    
    # Initializing architecture
    
    replay_fill = True
    wo_rec = True
    print("...filling Replay Buffer...")

 

    total_counter = 0
    scores = []
    for epoch in range(n_epochs):

        # Main process
        for episode in range(n_episodes):
            total_counter += 1
            global_obs, local_obs = env.reset()
           
            for step in range(n_steps):
                # Run the timestep
                illegal_moves = np.zeros(n_actions)
                illegal_moves = env.illegal_actions(illegal_moves)

                action = agent.choose_action(global_obs, local_obs, illegal_moves, replay_fill=replay_fill)
                next_gloabal_obs, next_local_obs, reward = env.step(action, rec_reward, rec_cons, rec_thres, without_rec=wo_rec)
                #env.render("Compare", realtime=True)

                # Save new information
                agent.store_transition(
                    global_obs, local_obs, next_gloabal_obs, next_local_obs, action, reward, illegal_moves)
                global_obs = next_gloabal_obs
                local_obs = next_local_obs

                agent.counter += 1
                

                if step % 4 == 0 and total_counter > episode_mem_size:
                    replay_fill = False #finish filling replay buffer
                    wo_rec = False
                    agent.learn()
            
            if total_counter % 150 == 0: print(total_counter)
    
            scores.append(1 - env.lastSim)
            
    
    print(f"score: {np.mean(scores[-50:])} g: {gamma}, ep: {epsilon}, alp: {alpha}, replace: {replace_target}, mem: {episode_mem_size}, rec_rew: {rec_reward}, rec_cons: {rec_cons}, rec_thres: {rec_thres} \n \n")
    return np.mean(scores[-50:] )



if __name__ == '__main__':
    # Parameter list to optimize
    parameters = {"gamma": (0.01, 0.99)}

    optimizer = BayesianOptimization(
        f=runner,
        pbounds=parameters,
        random_state=1,
    )

    optimizer.maximize(
        init_points=1,
        n_iter=10
    )

    print(optimizer.max)