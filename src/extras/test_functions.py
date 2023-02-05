import json
import numpy as np
from rich.progress import track

import logging

from models.Predictor import Predictor 
from data_statistics.Image_Generator import generate_image

from extras.logger import critical

log = logging.getLogger("tester")

@critical
def test_env(env, agent, data, n_episodes, n_steps=64, t_reward: bool = False, t_accuracy: bool = False, t_datarec : bool = False, t_speed : bool = False, t_vis: bool = False):
    #initialize
    predict = Predictor(mnist=True, emnist=True, quickdraw=True)
    data.shuffle()
    env.referenceData = data.pro_data
    ep_counter = 0

    reward_scores = []
    accuracy_scores = []
    datarec_scores = []
    speed_scores = []
    
    images = []
    image_indexes = iter(sorted(np.random.choice(n_episodes, 10, replace=False)) + [0])
    curInd = next(image_indexes)


    for episode in track(range(n_episodes), description="testing"):
        global_obs, local_obs = env.reset()
        score = 0
        done_step = 64
        
        for step in range(n_steps):
            # Run the timestep
            illegal_moves = np.zeros(env.n_actions)
            illegal_moves = env.illegal_actions(illegal_moves)

            env.curStep = step

            if not all(a == 1 for a in illegal_moves):
                    action = agent.choose_action(global_obs, local_obs, illegal_moves, replay_fill=False)
            else:
                action = np.random.choice(env.n_actions)


            # Check if the agent wants to stop at this current step
            if env.translate_action(action) == True:
                log.debug(f"AI is choosing the stop action in step: {step}")
                stop_step = step
                
                # Everything stays the same
                next_gloabal_obs = global_obs
                next_local_obs = local_obs
                
                # Calculate the new reward
                log.debug(f"Score before: {score}")
                reward = env.stop_reward(score=score, step=step)   
                log.info(f"stop_reward, {score},  {step}, {reward}")           
                log.debug(f"Stop reward: {reward}")
            else:
                # Draw further normally
                next_gloabal_obs, next_local_obs, reward = env.step(action, decrementor=1, rec_reward=0.1, without_rec=True, min_decrement=0.3)


            
            global_obs = next_gloabal_obs
            local_obs = next_local_obs
            

            if t_reward:
                score += reward
            if t_speed:
                if (1 - env.lastSim) > 0.75:
                    if data.dataset == "emnist":
                        rec = predict.emnist(env.reference) == predict.emnist(env.canvas)
                    elif data.dataset == "quickdraw":
                        rec = predict.quickdraw(env.reference) == predict.quickdraw(env.canvas)
                    else:
                        rec = predict.mnist(env.reference) == predict.mnist(env.canvas)
                    if rec and done_step == 64:
                        done_step = step

            if env.translate_action(action) == True:
                break


            
        if t_reward: 
            reward_scores.append(score)
        if t_accuracy: 
            accuracy_scores.append(1 - env.lastSim)
        if t_datarec:
            if data.dataset == "emnist":
                rec = predict.emnist(env.reference) == predict.emnist(env.canvas)
            elif data.dataset == "quickdraw":
                rec = predict.quickdraw(env.reference) == predict.quickdraw(env.canvas)
            else:
                rec = predict.mnist(env.reference) == predict.mnist(env.canvas)
            datarec_scores.append(int(rec))
        if t_speed:
            speed_scores.append(done_step)
        if t_vis:
                if ep_counter == curInd:
                    images.append(env.gradient_render())
                    curInd = next(image_indexes)
        
        ep_counter += 1
                
    scores = []
    if t_reward: scores.append(np.mean(reward_scores))
    if t_accuracy: scores.append(np.mean(accuracy_scores))
    if t_datarec: scores.append(np.mean(datarec_scores))
    if t_speed: scores.append(np.mean(speed_scores))
    if t_vis: generate_image(images, columns=2)
            
    return scores