import numpy as np
from rich.progress import track
import logging
from models.Predictor import Predictor 
from extras.logger import critical

log = logging.getLogger("tester")

@critical
def test_env(env, agent, data, n_episodes, n_steps=64, t_vis: bool = True):
    #initialize
    predict = Predictor(mnist=True, emnist=True, quickdraw=True)
    data.shuffle()

    reward_scores = []
    accuracy_scores = []
    datarec_scores = []
    speed_scores = []
    drawratio_scores = []
    overdraw_scores = []
    images = []
    image_indexes = iter(sorted(np.random.choice(n_episodes, 10, replace=False)) + [0])
    curInd = next(image_indexes)

    env.show_Reference = True

    for episode in track(range(n_episodes), description="testing"):
        global_obs, local_obs = env.reset()
        score = 0

        last_step = 0
        drawratio = 0

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
                log.info(f"Stopaction in step {step}")
                reward = env.stop_reward(score=score, step=step)
                next_gloabal_obs = global_obs  
                next_local_obs = local_obs         
            else:
                # Draw further normally
                next_gloabal_obs, next_local_obs, reward = env.step(action)
            
            global_obs = next_gloabal_obs
            local_obs = next_local_obs
            score += reward

            if env.isDrawing: 
                last_step = step
                drawratio += 1
            
            if env.translate_action(action) == True:
                break

        reward_scores.append(score)
        accuracy_scores.append(1 - env.lastSim)
        if data.dataset == "emnist":
            rec = env.label == predict.emnist(env.canvas, mode="test")
        elif data.dataset == "quickdraw":
            rec = env.label == predict.quickdraw(env.canvas, mode="test")
        else:
            rec = env.label == predict.mnist(env.canvas, mode="test")
        datarec_scores.append(int(rec))
        speed_scores.append(last_step)
        if last_step == 0: drawratio_scores.append(0)
        else: drawratio_scores.append(drawratio/last_step)
        overdraw_scores.append(env.overdrawn_perepisode)
        if t_vis:
            if episode == curInd:
                images.append(env.gradient_render())
                curInd = next(image_indexes)

    scores = []
    scores.append(np.mean(reward_scores))
    scores.append(np.mean(accuracy_scores))
    scores.append(np.mean(datarec_scores))
    scores.append(np.mean(speed_scores))
    scores.append(np.mean(drawratio_scores))
    scores.append(np.mean(overdraw_scores))
    scores.append(images)
    return scores



@critical
def generative_test_env(env, agent, n_episodes, n_steps=64):
    #initialize
    predict = Predictor(mnist=True, emnist=True, quickdraw=True)

    datarec_scores = []
    speed_scores = []
    drawratio_scores = []

    images = []
    image_indexes = iter(sorted(np.random.choice(n_episodes, 12, replace=False)) + [0])
    curInd = next(image_indexes)
    env.show_Reference = False
    
    for episode in track(range(n_episodes), description="testing"):
        global_obs, local_obs = env.reset()
        score = 0
        last_step = 0
        drawratio = 0
        
        for step in range(n_steps):
            # Run the timestep
            illegal_moves = np.zeros(env.n_actions)
            illegal_moves = env.illegal_actions(illegal_moves)

            env.curStep = step

            if not all(a == 1 for a in illegal_moves):
                    if agent.softmax:
                        action = agent.choose_action_softmax(global_obs, local_obs, illegal_moves, replay_fill=False)
                    else:
                        action = agent.choose_action(global_obs, local_obs, illegal_moves, replay_fill=False)
            else:
                action = np.random.choice(env.n_actions)

            # Check if the agent wants to stop at this current step
            if env.translate_action(action) == True:
                log.info(f"stopaction in step {step}")
                reward = env.stop_reward(step=step, score=score)  
                next_gloabal_obs = global_obs  
                next_local_obs = local_obs
            else:
                # Draw further normally
                next_gloabal_obs, next_local_obs, reward = env.step(action)

            if env.isDrawing: 
                last_step = step
                drawratio += 1

            global_obs = next_gloabal_obs
            local_obs = next_local_obs
            
            score += reward

            if env.translate_action(action) == True:
                break

        rec = env.label == predict.mnist(env.canvas, mode="test")
        datarec_scores.append(int(rec))
        speed_scores.append(last_step)
        if last_step == 0: drawratio_scores.append(0)
        else: drawratio_scores.append(drawratio/last_step)
        if episode == curInd:
            images.append(env.gradient_render()[1])
            curInd = next(image_indexes)
                
    scores = []
    scores.append(np.mean(datarec_scores))
    scores.append(np.mean(speed_scores))
    scores.append(np.mean(drawratio_scores))
    scores.append(images)
    return scores