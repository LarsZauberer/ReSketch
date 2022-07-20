from agent_modules.environment import ShapeDraw
from agent_modules.nn_agent import DeepQNetwork, Agent
from data.ai_data import AI_Data
from result_stats.learn_plotter import Learn_Plotter
import numpy as np
import json

if __name__ == '__main__':
    # Hyper parameters
    canvas_size = 28
    patch_size = 7
    n_actions = 2*(patch_size**2)
    episode_mem_size = 200
    batch_size = 64
    n_episodes = 4000
    n_steps = 64
    n_epochs = 3

    # further calculations
    glob_in_dims = (4, canvas_size, canvas_size)
    loc_in_dims = (2, patch_size, patch_size)
    mem_size = episode_mem_size*n_steps

    #load Data
    learn_plot = Learn_Plotter(path="src/result_stats/plotlearn_data.json")
    data = AI_Data(path="src/data/train_ref_Data.json")
    data.sample(n_episodes)
    env = ShapeDraw(canvas_size, patch_size, data.pro_data)
    agent_args = {"gamma": 0.99, "epsilon": 1, "alpha": 0.001, "replace_target": 1000, 
                  "global_input_dims": glob_in_dims, "local_input_dims": loc_in_dims, 
                  "mem_size": mem_size, "batch_size": batch_size, 
                  "q_next_dir": "src/nn_memory/q_next", "q_eval_dir": "src/nn_memory/q_eval"}
    agent = Agent(**agent_args)
    
    # Initializing architecture
    

    print("...filling Replay Buffer...")
    total_counter = 0
    scores = []
    for epoch in range(n_epochs):
        data.shuffle()
        env.referenceData = data.pro_data
        
        # Main process
        for episode in range(n_episodes):
            total_counter += 1
            global_obs, local_obs = env.reset()
            score = 0

            for step in range(n_steps):
                # Run the timestep
                action = agent.choose_action(global_obs, local_obs)
                next_gloabal_obs, next_local_obs, reward = env.step(action)
                #env.render("Compare", realtime=True)

                # Save new information
                agent.store_transition(
                    global_obs, local_obs, next_gloabal_obs, next_local_obs, action, reward)
                global_obs = next_gloabal_obs
                local_obs = next_local_obs

                agent.counter += 1
                score += reward

                if step % 4 == 0 and total_counter > episode_mem_size:
                    agent.learn()
                    agent.epsilon = 0
            

            # Learn Process visualization
            if total_counter > 200:
                if total_counter % 12 == 0:
                    avg_score = np.mean(scores)
                    scores = []
                    print(f"episode: {total_counter}, score: {score}, average score: {'%.3f' % avg_score}, epsilon: {'%.3f' % agent.epsilon}")

                    env.render("Compare")
                    learn_plot.update_plot(total_counter, avg_score)
                else:
                    print(f"episode: {total_counter}, score: {score}")

                scores.append(score)
            
    
        #save weights
        agent.save_models()
        learn_plot.save_plot()


agent.save_models()
learn_plot.save_plot()



