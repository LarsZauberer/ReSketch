import argparse
import json
import numpy as np

def hyperparameter_loader(path, modelName):
    # Load Hyperparameter data
    with open(path, "r") as f:
        hyp_data = json.load(f)
    if hyp_data.get(modelName, False):
        #General model names: "base", "mnist", "speed" "mnist-speed"
        hyp_data = hyp_data[modelName]
    else:
        #No known Model: give default Values
        hyp_data = {"gamma": 0.7, "epsilon": 0, "alpha": 0.0002, "replace_target": 6000, "episode_mem_size": 100, "n_episodes": 200} 
    
    return hyp_data






def train(env, agent, data, learn_plot, episode_mem_size, n_episodes, n_steps, model_path, save_training=True, vis_compare=12, mnist=False, speed=False):
    # Initializing architecture
    total_counter = 0
    scores = []
    
    #prepare Data
    data.shuffle()
    env.referenceData = data.pro_data

    #Fill Replay Buffer
    no_rec = True
    replay_fill = True
    print("...filling Replay Buffer...")

    # Main process
    for episode in range(n_episodes):


       

        if not replay_fill: env.curEpisode += 1
        total_counter += 1
        global_obs, local_obs = env.reset()
        score = 0

        done_step = None
        done_accuracy = 0.25 + 0.5*(total_counter/n_episodes)

        for step in range(n_steps):
            # Run the timestep
            illegal_moves = np.zeros(env.n_actions)
            illegal_moves = env.illegal_actions(illegal_moves)
            env.curStep = step

            action = agent.choose_action(global_obs, local_obs, illegal_moves, replay_fill=replay_fill)
            next_gloabal_obs, next_local_obs, reward = env.step(action, decrementor=n_episodes-episode_mem_size, rec_reward=0.1, min_decrement=0.3, without_rec=no_rec)
            #env.render("Compare", realtime=True)

            if done_step == None and not replay_fill and speed: 
                if env.agent_is_done(done_accuracy): done_step = step
            
            # Save new information
            agent.store_transition(
                global_obs, local_obs, next_gloabal_obs, next_local_obs, action, reward, illegal_moves)
            global_obs = next_gloabal_obs
            local_obs = next_local_obs

            agent.counter += 1
            score += reward

            if step % 4 == 0 and total_counter > episode_mem_size:
                no_rec = not mnist
                replay_fill = False #finish filling replay buffer
                agent.learn()
        
        if speed:
            speed_reward = env.speed_reward(done_step)
            #if not replay_fill: print(speed_reward)
            agent.update_speedreward(speed_reward)
            

    
        # Learn Process visualization
        if total_counter > episode_mem_size:
            real_ep = total_counter - episode_mem_size
            if real_ep % abs(vis_compare) == 0:
                avg_score = np.mean(scores)
                scores = []
                print(f"episode: {real_ep}, score: {score}, average score: {'%.3f' % avg_score}, epsilon: {'%.3f' % agent.epsilon}")
                if vis_compare > 0: env.render()
                learn_plot.update_plot(real_ep, avg_score)
            else:
                print(f"episode: {real_ep}, score: {score}")

            scores.append(score)


    if save_training:
        # save weights
        agent.save_models(model_path)
        learn_plot.save_plot()