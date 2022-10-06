from threading import activeCount
from agent_modules.environment import ShapeDraw
from agent_modules.nn_agent import DeepQNetwork, Agent
from data.ai_data import AI_Data
from result_stats.learn_plotter import Learn_Plotter
import numpy as np
import json
from time import sleep


def train(env, agent, data, learn_plot, n_episodes, n_epochs, n_steps, n_actions, episode_mem_size, save_training=True, vis_compare=12, rec_reward=0.1, min_decrement=0.3, mnist=False, speed=False):
    # Initializing architecture
    wo_rec = True
    replay_fill = True
    print("...filling Replay Buffer...")

    total_counter = 0
    scores = []
    for epoch in range(n_epochs):
        data.shuffle()
        env.referenceData = data.pro_data

        # Main process
        for episode in range(n_episodes):

            if not replay_fill: env.curEpisode += 1
            total_counter += 1
            global_obs, local_obs = env.reset()
            score = 0
            done_step = None

            done_accuracy = 0.3 + 0.5*(total_counter/n_episodes)

            for step in range(n_steps):
                # Run the timestep
                illegal_moves = np.zeros(n_actions)
                illegal_moves = env.illegal_actions(illegal_moves)
                env.curStep = step

                if not all(a == 1 for a in illegal_moves):
                    action = agent.choose_action(global_obs, local_obs, illegal_moves, replay_fill=replay_fill)
                else:
                    action = np.random.choice(n_actions)
                    

                next_gloabal_obs, next_local_obs, reward = env.step(action, decrementor=n_episodes-episode_mem_size, rec_reward=rec_reward, min_decrement=min_decrement, without_rec=wo_rec)

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
                    wo_rec = not mnist
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
            agent.save_models()
            learn_plot.save_plot()
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mnist", help="Weather to use mnist training or not", action="store_true", default=False)
    parser.add_argument("-s", "--speed", help="Weather to use speed training or not", action="store_true", default=False)
    args = parser.parse_args()

    # Options
    mnist = args.mnist
    speed = args.speed
    
    # Load Hyperparameter data
    with open("src_physics/opti.json", "r") as f:
        hyp_data = json.load(f)
        
    if mnist and speed: model ="mnist-speed"
    elif mnist: model = "mnist"  
    elif speed: model = "speed"
    else: model = "base"
    hyp_data = hyp_data[model]
    
    print(f"Hyperparameters: {hyp_data}")
    
    # Hyper parameters
    canvas_size = 28
    patch_size = 5
    n_actions = 42
    episode_mem_size = int(hyp_data["episode_mem_size"])
    batch_size = 64
    n_episodes = int(hyp_data["n_episodes"]) + episode_mem_size
    n_steps = 64
    n_epochs = 1
    max_action_strength = 1

    # Mnist Train parameters
    min_decrement = hyp_data.get("min_decrement", 0.3)
    rec_reward = hyp_data.get("rec_reward", 0.1)
    


    # further calculations
    glob_in_dims = (4, canvas_size, canvas_size)
    loc_in_dims = (2, patch_size, patch_size)
    mem_size = episode_mem_size*n_steps

    # load Data
    learn_plot = Learn_Plotter(path="src_physics/result_stats/plotlearn_data.json")
    data = AI_Data(dataset="mnist")
    data.sample(n_episodes)

    env = ShapeDraw(canvas_size, patch_size, data.pro_data, n_actions=n_actions, max_action_strength=max_action_strength, friction=hyp_data["friction"], vel_1=hyp_data["vel_1"], vel_2=hyp_data["vel_2"])
    agent_args = {"gamma": hyp_data["gamma"], "epsilon": hyp_data["epsilon"], "alpha": hyp_data["alpha"], "replace_target": int(hyp_data["replace_target"]), 
                  "global_input_dims": glob_in_dims, "local_input_dims": loc_in_dims, 
                  "mem_size": mem_size, "batch_size": batch_size, "model": f"physics-{model}", "n_actions": n_actions}
    agent = Agent(**agent_args)
    
    # Start training
    train(env=env,
          agent=agent,
          data=data,
          learn_plot=learn_plot,
          n_steps=n_steps,
          n_episodes=n_episodes,
          n_epochs=n_epochs,
          n_actions=n_actions,
          episode_mem_size=episode_mem_size,
          save_training=True,
          vis_compare=-12,
          rec_reward=rec_reward,
          min_decrement=min_decrement,
          mnist=mnist,
          speed=speed,
          )
