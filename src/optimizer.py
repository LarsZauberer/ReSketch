from dataclasses import replace
from agent_modules.environment import ShapeDraw
from agent_modules.nn_agent import DeepQNetwork, Agent
from data.ai_data import AI_Data
from result_stats.learn_plotter import Learn_Plotter
import numpy as np
import json
from bayes_opt import BayesianOptimization
from rich.progress import track


# constant parameters
canvas_size = 28
patch_size = 5
n_actions = 42
episode_mem_size = 200
batch_size = 64
n_episodes = 1000
n_steps = 64
n_epochs = 1
max_action_strength = 1


def runner(gamma, epsilon, alpha, episode_mem_size, replace_target, friction, vel_1, vel_2):
    episode_mem_size = int(episode_mem_size)
    replace_target = int(replace_target)

    # further calculations
    glob_in_dims = (4, canvas_size, canvas_size)
    loc_in_dims = (2, patch_size, patch_size)
    mem_size = episode_mem_size*n_steps

    data = AI_Data(path="src/data/train_ref_Data.json")
    data.sample(n_episodes)
    env = ShapeDraw(canvas_size, patch_size, data.pro_data,
                    max_action_strength=max_action_strength, n_actions=n_actions, friction=friction, vel_1=vel_1, vel_2=vel_2)
    agent_args = {"gamma": gamma, "epsilon": epsilon, "alpha": alpha, "replace_target": replace_target,
                  "global_input_dims": glob_in_dims, "local_input_dims": loc_in_dims,
                  "mem_size": mem_size, "batch_size": batch_size,
                  "q_next_dir": "src/nn_memory/q_next", "q_eval_dir": "src/nn_memory/q_eval", "n_actions": n_actions}
    agent = Agent(**agent_args)

    replay_fill = True
    total_counter = 0
    scores = []
    for epoch in range(n_epochs):
        # ! Don't shuffle to get a better comparison
        env.referenceData = data.pro_data

        # Main process
        for episode in track(range(n_episodes), description="Running Iteration"):
            total_counter += 1
            global_obs, local_obs = env.reset()
            score = 0

            for step in range(n_steps):
                # Run the timestep
                illegal_moves = np.zeros(n_actions)
                illegal_moves = env.illegal_actions(illegal_moves)

                if not all(a == 1 for a in illegal_moves):
                    action = agent.choose_action(global_obs, local_obs, illegal_moves, replay_fill=replay_fill)
                else:
                    action = np.random.choice(n_actions)
                    
                next_gloabal_obs, next_local_obs, reward = env.step(action)

                # Save new information
                agent.store_transition(
                    global_obs, local_obs, next_gloabal_obs, next_local_obs, action, reward, illegal_moves)
                global_obs = next_gloabal_obs
                local_obs = next_local_obs

                agent.counter += 1
                score += reward

                if step % 4 == 0 and total_counter > episode_mem_size:
                    replay_fill = False  # finish filling replay buffer
                    agent.learn()

            # Save the score
            scores.append(score)
            # print("Episode: {}, Score: {}".format(episode, score))

            if total_counter > 200:
                if total_counter % 12 == 0:
                    avg = np.mean(scores[-12:])
                    print("Avg: ", avg)

    print("Iteration complete: {}".format(np.mean(scores[-12:])))
    return np.mean(scores[-12:])


if __name__ == '__main__':
    # Parameter list to optimize
    parameters = {"gamma": (0.001, 0.99), "epsilon": (
        0.1, 1), "alpha": (0.0001, 0.001),
        "episode_mem_size": (200, 1000), "replace_target": (1000, 10000), "friction": (0.01, 0.6), "vel_1": (0.1, 1), "vel_2": (0.1, 2)}

    optimizer = BayesianOptimization(
        f=runner,
        pbounds=parameters,
        random_state=1,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=10,
    )

    print(optimizer.max)
