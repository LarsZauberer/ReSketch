import numpy as np
import logging
from rich.progress import Progress
from extras.logger import critical

log = logging.getLogger("trainer")


@critical
def train(env, agent, data, learn_plot, episode_mem_size, n_episodes, n_steps, model_path, save_training=True, vis_compare=12):
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
    """

    # Initializing architecture
    scores = []
    env.show_Reference = True
    
    #prepare Data
    data.shuffle()
    env.referenceData = data.pro_data

    #Fill Replay Buffer
    replay_fill = True
    log.info("...filling Replay Buffer...")
    progress = Progress()
    progress.__enter__()
    replay_fill_task = progress.add_task("[red]Replay Buffer Filling", total=episode_mem_size)

    # Main process
    for episode in range(n_episodes):

        env.show_Reference = True
        if env.generative:
            rand = np.random.random()
            thres = 1 - 1.5*(episode-episode_mem_size)/(n_episodes-episode_mem_size)
            if thres < 0.30: thres = 0.30
            agent.set_softmax_temp(0.05 + (1-thres)*0.08)
            if rand > thres:
                env.show_Reference = False
            else:
                env.show_Reference = True

        progress.update(replay_fill_task, advance=1)
        global_obs, local_obs = env.reset()
        score = 0

        for step in range(n_steps):
             # Run the timestep
            illegal_moves = np.zeros(env.n_actions)
            illegal_moves = env.illegal_actions(illegal_moves)
            env.curStep = step

            # Choose Action
            if not all(a == 1 for a in illegal_moves):
                action = agent.choose_action_softmax(global_obs, local_obs, illegal_moves, replay_fill=replay_fill)
                #action = agent.choose_action(global_obs, local_obs, illegal_moves, replay_fill=replay_fill)
                
                """ if env.show_Reference:
                    action = agent.choose_action(global_obs, local_obs, illegal_moves, replay_fill=replay_fill)
                else:
                    action = agent.choose_action_softmax(global_obs, local_obs, illegal_moves, replay_fill=replay_fill) """
            else:
                action = np.random.choice(env.n_actions)

            
            

             # Make Step
            if env.translate_action(action) == True:
                #Agent chooses stop-action
                if env.generative:
                    reward = env.generative_stop_reward(step=step, score=score)
                else:
                    reward = env.stop_reward(score=score, step=step) 
                log.info(f"stopAction in Step: {step}, Accuracy: {score}, stopReward: {reward}")   
            else:
                # Draw further normally
                next_gloabal_obs, next_local_obs, reward = env.step(score, action)

            

            # Save step information
            agent.store_transition(global_obs, local_obs, next_gloabal_obs, next_local_obs, action, reward, illegal_moves)

            global_obs = next_gloabal_obs
            local_obs = next_local_obs
            score += reward

            

            #learn
            agent.update_graph()
            if step % 4 == 0 and episode > episode_mem_size:
                replay_fill = False #finish filling replay buffer
                agent.learn()
            
            #Stop by StopAction
            if env.translate_action(action) == True:
                break
            
        # Learn Process visualization
        if episode > episode_mem_size:
            agent.reduce_epsilon()

            progress.update(replay_fill_task, visible=False)

            real_ep = episode - episode_mem_size
            if real_ep % abs(vis_compare) == 0:
                avg_score = np.mean(scores)
                scores = []
                log.info(f"episode: {real_ep}, score: {score}, average score: {'%.3f' % avg_score}, epsilon: {'%.3f' % agent.epsilon}")
                if vis_compare > 0: env.render()
                learn_plot.update_plot(real_ep, avg_score)
            else:
                log.info(f"episode: {real_ep}, score: {score}, showref: {env.show_Reference}")

            scores.append(score)


    #finish Training
    log.info("training completed")
    if save_training:
        # save weights
        agent.save_models(model_path)
        learn_plot.save_plot()
   
    