import argparse
import json
import numpy as np
from pathlib import Path
import logging
from rich.progress import Progress

from extras.logger import critical

log = logging.getLogger("trainer")


@critical
def train(env, agent, data, learn_plot, episode_mem_size, n_episodes, n_steps, model_path, save_training=True, vis_compare=12, mnist=False, speed=False):
    """
    train Trains a specific model in an environment

    :param env: The environment where the agent is moving
    :type env: environment.Environment
    :param agent: The agent moving in the environment
    :type agent: nn_agent.Agent
    :param data: The data for the environment
    :type data: data.ai_data.Ai_Data
    :param learn_plot: The matplotlib plot where to plot the learning curve
    :type learn_plot: matplotlib.pyplot.Figure
    :param episode_mem_size: How big the replay buffer is allowed to be
    :type episode_mem_size: int
    :param n_episodes: How many episodes to train the model
    :type n_episodes: int
    :param n_steps: Allowed number of steps the agent is allowed to take per episode
    :type n_steps: int
    :param model_path: The path to the file where the checkpoint files of the model should be saved
    :type model_path: str
    :param save_training: Should this training be saved to a checkpoint file, defaults to True
    :type save_training: bool, optional
    :param vis_compare: How often the visual comparison plot should be shown to the developer, defaults to 12
    :type vis_compare: int, optional
    :param mnist: Is this a mnist variation, defaults to False
    :type mnist: bool, optional
    :param speed: Is this a speed variation, defaults to False
    :type speed: bool, optional
    """
    # Initializing architecture
    total_counter = 0
    scores = []
    
    #prepare Data
    data.shuffle()
    env.referenceData = data.pro_data

    #Fill Replay Buffer
    no_rec = True
    replay_fill = True
    log.info("...filling Replay Buffer...")

    progress = Progress()
    progress.__enter__()
    replay_fill_task = progress.add_task("[red]Replay Buffer Filling", total=episode_mem_size)

    never_ask_again = False

    # Main process
    for episode in range(n_episodes):
        if not replay_fill: env.curEpisode += 1
        total_counter += 1
        progress.update(replay_fill_task, advance=1)
        global_obs, local_obs = env.reset()
        score = 0

        done_step = None
        done_accuracy = 0.25 + 0.5*(total_counter/n_episodes)
        
        per_episode_replay_buffer = []
        stop_step = -1
        dont_ask_again = False

        for step in range(n_steps):
            """ if stop_step >= 0:
                try:
                    agent.store_transition(*per_episode_replay_buffer[(step - stop_step + 1) % len(per_episode_replay_buffer)])  # Repeat the drawing process in the replay buffer till it has reaches 64
                    continue
                except Exception:
                    pass """
            
            # Run the timestep
            illegal_moves = np.zeros(env.n_actions)
            illegal_moves = env.illegal_actions(illegal_moves)
            env.curStep = step

            if not all(a == 1 for a in illegal_moves):
                action = agent.choose_action(global_obs, local_obs, illegal_moves, replay_fill=replay_fill)
            else:
                action = np.random.choice(env.n_actions)
            
            # Supervised stopping
            weight = 1
            if score > 0.5 and not replay_fill and total_counter - episode_mem_size < 200 and not dont_ask_again and not never_ask_again:
                env.render()
                log.debug(f"Triggered supervised help in step: {step}")
                inp = input("> ")
                if inp == "1":
                    dont_ask_again = True
                elif inp == "2":
                    pass
                elif inp == "3":
                    action = 98  # Stop action
                    weight = 1
                elif inp == "4":
                    action = 98  # Stop action
                    weight = 10
                elif inp == "q":
                    never_ask_again = True

            # Check if the agent wants to stop at this current step
            if env.translate_action(action) == True:
                log.debug(f"AI is choosing the stop action in step: {step}")
                stop_step = step
                
                # Everything stays the same
                next_gloabal_obs = global_obs
                next_local_obs = local_obs
                
                # Calculate the new reward
                log.debug(f"Score before: {score}")
                reward = env.stop_reward(score=score, step=step, weight=weight)
                log.debug(f"Stop reward: {reward}")
            else:
                # Draw further normally
                next_gloabal_obs, next_local_obs, reward = env.step(action, decrementor=n_episodes-episode_mem_size, rec_reward=0.1, min_decrement=0.3, without_rec=no_rec)

            if done_step == None and not replay_fill and speed: 
                if env.agent_is_done(done_accuracy): done_step = step
            
            # Save new information
            agent.store_transition(
                global_obs, local_obs, next_gloabal_obs, next_local_obs, action, reward, illegal_moves)
            per_episode_replay_buffer.append((global_obs, local_obs, next_gloabal_obs, next_local_obs, action, reward, illegal_moves))
            global_obs = next_gloabal_obs
            local_obs = next_local_obs

            agent.counter += 1
            score += reward

            if step % 4 == 0 and total_counter > episode_mem_size:
                no_rec = not mnist
                replay_fill = False #finish filling replay buffer
                agent.learn()
                
            if env.translate_action(action) == True:
                break
        
        if speed:
            speed_reward = env.speed_reward(done_step)
            #if not replay_fill: print(speed_reward)
            agent.update_speedreward(speed_reward)
            
            
        # Learn Process visualization
        if total_counter > episode_mem_size:
            progress.update(replay_fill_task, visible=False)
            real_ep = total_counter - episode_mem_size
            if real_ep % abs(vis_compare) == 0:
                avg_score = np.mean(scores)
                scores = []
                log.info(f"episode: {real_ep}, score: {score}, average score: {'%.3f' % avg_score}, epsilon: {'%.3f' % agent.epsilon}")
                if vis_compare > 0: env.render()
                learn_plot.update_plot(real_ep, avg_score)
            else:
                log.info(f"episode: {real_ep}, score: {score}")

            scores.append(score)

    if save_training:
        # save weights
        agent.save_models(model_path)
        learn_plot.save_plot()
    
    progress.__exit__()
    