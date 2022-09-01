from agent_modules.environment import ShapeDraw
from agent_modules.nn_agent import DeepQNetwork, Agent
from data.ai_data import AI_Data
from result_stats.learn_plotter import Learn_Plotter, Traintest_Learn_Plotter
from nn_test import Test_NN
import numpy as np
import json

if __name__ == '__main__':
    # Hyper parameters
    canvas_size = 28
    patch_size = 7
    n_actions = 2*(patch_size**2)
    episode_mem_size = 600
    batch_size = 64
    n_episodes = 4000
    n_steps = 64
    n_epochs = 1

    # further calculations
    glob_in_dims = (4, canvas_size, canvas_size)
    loc_in_dims = (2, patch_size, patch_size)
    mem_size = episode_mem_size*n_steps

    #load Data
    learn_plot = Traintest_Learn_Plotter(path="src/result_stats/plotlearn_data.json")
    data = AI_Data(path="src/data/train_ref_Data.json")
    testing = Test_NN(n_test=24)
    

    data.pro_data = [np.reshape(data.ref_data[2][45], (28,28))]


    env = ShapeDraw(canvas_size, patch_size, data.pro_data)
    agent_args = {"gamma": 0.8, "epsilon": 0.2, "alpha": 0.0004545815846602133, "replace_target": 4685, 
                  "global_input_dims": glob_in_dims, "local_input_dims": loc_in_dims, 
                  "mem_size": mem_size, "batch_size": batch_size, 
                  "q_next_dir": "src/nn_memory/q_next", "q_eval_dir": "src/nn_memory/q_eval"}
    agent = Agent(**agent_args)
    

    # Initializing architecture
    wo_rec = True
    replay_fill = True
    print("...filling Replay Buffer...")

    total_counter = 0
    scores = []
    predicts = []
    sims = []
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
                illegal_moves = np.zeros(n_actions)
                illegal_moves = env.illegal_actions(illegal_moves)

                action = agent.choose_action(global_obs, local_obs, illegal_moves, replay_fill=replay_fill)
                next_gloabal_obs, next_local_obs, reward = env.step(action,  decrementor=3400, rec_reward=0.1, without_rec=wo_rec)
                #env.render("Compare", realtime=True)

                # Save new information
                agent.store_transition(
                    global_obs, local_obs, next_gloabal_obs, next_local_obs, action, reward, illegal_moves)
                global_obs = next_gloabal_obs
                local_obs = next_local_obs

                agent.counter += 1
                score += reward

                if step % 4 == 0 and total_counter > episode_mem_size:
                    wo_rec = False
                    replay_fill = False #finish filling replay buffer
                    agent.learn()

                

            # Learn Process visualization
            if total_counter > episode_mem_size:
                real_ep = total_counter - episode_mem_size

                

                if real_ep % 12 == 0:
                    avg_score = np.mean(scores)
                    avg_predicts = np.mean(predicts)
                    avg_sims = np.mean(sims)
                    test_score = testing.mnist_test(agent)

                    predicts = []
                    sims = []
                    scores = []
                    print(f"episode: {real_ep}, score: {score}, average score: {'%.3f' % avg_score} test: {'%.3f' % test_score}, predictions: {'%.3f' % avg_predicts} sims: {'%.3f' % avg_sims} epsilon: {'%.3f' % agent.epsilon}")

                    #env.render("Compare")
                    learn_plot.update_plot(real_ep, avg_score, test_score)
                else:
                    print(f"episode: {real_ep}, score: {score}")

                scores.append(score)
                ref, canv = env.predict_mnist()
                predicts.append(1 if ref == canv else 0)
                sims.append(1-env.lastSim)

            
            
        #save weights
        agent.save_models()
        learn_plot.save_plot()















