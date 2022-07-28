import os
import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, Input, concatenate
from tensorflow.keras.utils import plot_model

import numpy as np
from time import sleep



class DeepQNetwork(object):
    def __init__(self, lr, n_actions: int, batch_size: int, name: str,
                 global_input_dims: int, local_input_dims: int, fc1_dims: int = 512, chkpt_dir='tmp/dqn'):
        self.lr = lr  # The optimization learning rate of the network model
        self.n_actions = n_actions  # How many actions the agent has available -> Index of the action to execute
        self.name = name  # The identification name of the agent

        # Inputs
        self.global_input_dims = global_input_dims  # The dimensions of the global input. So the dimensions of the full drawing canvas.
        self.local_input_dims = local_input_dims  # The dimensions of the concentrated input. The local patch of the canvas.
        self.fc1_dims = fc1_dims  # Dimensions of the last dense layer
        self.batch_size = batch_size  # How many inputs sending into the network

        # saving / memory
        self.checkpoint_file = os.path.join(chkpt_dir, 'deepqnet.ckpt')  # Where the model should be saved

        # Generate the network
        self.build_network()

    def build_network(self):
        """
        build_network Generate a keras DeepQNetwork. The model will be saved in the self.dqn variable.
        """
        # global convolution
        glob_in = Input(shape=self.global_input_dims,
                        batch_size=self.batch_size, name="global_input")
        glob_conv1 = Conv2D(32, (8, 8), strides=3,  activation="relu", input_shape=self.global_input_dims,
                            padding="same", name="glob_conv1", data_format='channels_first')(glob_in)
        glob_conv2 = Conv2D(64, (4, 4), strides=2, activation="relu", name="glob_conv2",
                            padding="same", data_format='channels_first')(glob_conv1)
        glob_conv3 = Conv2D(64, (3, 3), strides=1, activation="relu", name="glob_conv3",
                            padding="same", data_format='channels_first')(glob_conv2)

        # local convolution
        loc_in = Input(shape=self.local_input_dims,
                       name="local_input", batch_size=self.batch_size)
        loc_conv1 = Conv2D(128, (11, 11), strides=1, activation="relu",
                           name="loc_conv1", padding="same", data_format='channels_first')(loc_in)

        # concat
        concat_model = concatenate([glob_conv3, loc_conv1], axis=1)

        # Fully connected Layers
        concat_model = Flatten(
            name="Flatten", data_format="channels_first")(concat_model)
        dense1 = Dense(self.fc1_dims, activation="relu",
                       name="dense1")(concat_model)
        out = Dense(self.n_actions, activation="relu", name="output")(dense1)

        # Inputs of the network are global and local states: glob_in = [4x28x28], loc_in = [2x7x7]
        # Output of the netword are Q-values. each Q-value represents an action
        self.dqn = Model(inputs=[glob_in, loc_in], outputs=[out])
        

        # Network is ready for calling / Training
        self.dqn.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.lr), metrics=["accuracy"])

        plot_model(self.dqn, to_file=f"{self.name}.png", show_shapes=True)

        #calling: dqn([global_state_batch, local_state_batch])
        #training: dqn.train_on_batch(x=[global_state_batch, local_state_batch], y=q_target)

    def load_checkpoint(self):
        """
        load_checkpoint Load a network checkpoint from the file
        """
        print("...Loading checkpoint...")
        self.dqn.load_weights(self.checkpoint_file)

    def save_checkpoint(self):
        """
        save_checkpoint Save a network checkpoint to the file
        """
        print("...Saving checkpoint...")
        self.dqn.save_weights(self.checkpoint_file)


    


class Agent(object):
    def __init__(self, alpha, gamma, mem_size, epsilon, global_input_dims, local_input_dims, batch_size,
                 replace_target=1000,
                 q_next_dir='src/nn_memory/q_next', q_eval_dir='src/nn_memory/q_eval'):

        self.n_actions = local_input_dims[0]*(local_input_dims[1]**2)  # How many action options the agent has. -> Index of the action to choose
        self.action_space = [i for i in range(self.n_actions)]  # All the actions the agent can choose
        self.gamma = gamma  # Is the learnrate
        self.mem_size = mem_size  # The allocated memory size (The number of slots for saved observation)
        self.counter = 0  # Counter of every step
        self.epsilon = epsilon  # The epsilon value of the agent. The exploration value
        self.batch_size = batch_size  # How big each batch of inputs is
        self.replace_target = replace_target  # When to update the q_next network
        self.global_input_dims = global_input_dims  # The input dimensions of the whole canvas.
        self.local_input_dims = local_input_dims  # The dimensions of the concentrated patch of the canvas

        self.q_next = DeepQNetwork(alpha, self.n_actions, self.batch_size, global_input_dims=global_input_dims,
                                   local_input_dims=local_input_dims, name='q_next', chkpt_dir=q_next_dir)  # The QNetwork to compute the q-values on the next state of the canvas
        self.q_eval = DeepQNetwork(alpha, self.n_actions, self.batch_size, global_input_dims=global_input_dims,
                                   local_input_dims=local_input_dims, name='q_eval', chkpt_dir=q_eval_dir) # The QNetwork to compute the q-values on the current state of the canvas

        # Dimensions of Replay buffer memory
        glob_mem_shape = (
            self.mem_size, global_input_dims[0], global_input_dims[1], global_input_dims[2])
        loc_mem_shape = (
            self.mem_size, local_input_dims[0], local_input_dims[1], local_input_dims[2])
        illegal_list_shape = (self.mem_size, self.n_actions)
        
        # Replay buffer
        self.global_state_memory = np.zeros(glob_mem_shape)
        self.local_state_memory = np.zeros(loc_mem_shape)
        self.new_global_state_memory = np.zeros(glob_mem_shape)
        self.new_local_state_memory = np.zeros(loc_mem_shape)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.illegal_list_memory = np.zeros(illegal_list_shape)

        self.recent_mem = 10
        self.recent_actions = np.zeros(self.recent_mem)

    def store_transition(self, global_state: np.array, local_state: np.array, next_gloabal_state: np.array, next_local_state: np.array, action: int, reward: float, illegal_list : np.array):
        """
        store_transition Save the next step to the replay buffer

        :param global_state: The global state of the canvas
        :type global_state: np.array
        :param local_state: The small patch of the canvas
        :type local_state: np.array
        :param next_gloabal_state: The next state of the whole canvas
        :type next_gloabal_state: np.array
        :param next_local_state: The next state of the small patch of the canvas
        :type next_local_state: np.array
        :param action: The number representing the action the Agent took
        :type action: int
        :param reward: The reward the agent got
        :type reward: float
        """
        index = self.counter % self.mem_size

        self.global_state_memory[index] = global_state
        self.local_state_memory[index] = local_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.illegal_list_memory[index] = illegal_list
        self.new_global_state_memory[index] = next_gloabal_state
        self.new_local_state_memory[index] = next_local_state



    def choose_action(self, global_state: np.array, local_state: np.array, illegal_list : np.array, replay_fill: bool = False):
        """
        choose_action Agent should choose an action from the action_space

        :param global_state: The whole canvas
        :type global_state: np.array
        :param local_state: The small patch of the canvas
        :type local_state: np.array
        :return: Index of the action the agent wants to take
        :rtype: int
        """
        action = 0

        # Check if the agent should explore
        rand = np.random.random()
        if rand < self.epsilon or self.rare_Exploration() or replay_fill:
            while True:
                action = np.random.choice(self.action_space)
                if illegal_list[action] == 1:
                    continue
                    
                else: break
        else:
            if self.counter % self.replace_target == 0 and self.counter > 0:
                # Updates the q_next network. closes the gap between q_eval and q_next to avoid q_next getting outdated
                self.update_graph()
            # create batch of states (prediciton must be in batches)
            # Create a batch containing only one real state (all zeros for the other states)

            glob_batch = np.array([global_state])
            loc_batch = np.array([local_state])
            """ for _ in range(self.batch_size-1):
                glob_batch = np.append(glob_batch, np.array(
                    [np.zeros(self.global_input_dims)]), axis=0)
                loc_batch = np.append(loc_batch, np.array(
                    [np.zeros(self.local_input_dims)]), axis=0) """

            # Ask the QNetwork for an action
            actions = np.array(self.q_eval.dqn([glob_batch, loc_batch])[0])

            while illegal_list[np.argmax(actions)] == 1:
                
                actions[np.argmax(actions)] = -1

            # Take the index of the maximal value -> action
            action = int(np.argmax(actions))

        # Only important for the rare_exploration
        action_ind = self.counter % self.recent_mem
        self.recent_actions[action_ind] = action


        return action

    def learn(self):
        """
        learn the Training of The agent/network. Based on deep Q-learning
        """
        

        # randomly samples Memory.
        # chooses as many states from Memory as batch_size requires
        max_mem = self.counter if self.counter < self.mem_size else self.mem_size
        # Get random state inputs from the replay buffer
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        # batch = [0, 5, 11, ..., batch_size] type: np.array

        # load sampled memory
        global_state_batch = self.global_state_memory[batch]
        local_state_batch = self.local_state_memory[batch]
        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        new_global_state_batch = self.new_global_state_memory[batch]
        new_local_state_batch = self.new_local_state_memory[batch]
        illegal_list_batch = self.illegal_list_memory[batch]

        # runs network -> delivers output for training
        # gives the outpus (Q-values) of current states and next states. 
        # It gives this output of every state in the batch
        # type: np.array example: [ [0.23, 0.64, 0.33, ..., n_actions], ..., batch_size]


        q_eval = np.array(self.q_eval.dqn([global_state_batch, local_state_batch]))
        q_next = np.array(self.q_next.dqn([new_global_state_batch, new_local_state_batch]))

        # Calculates optimal output for training. ( Bellman Equation !! )



        q_target = np.copy(q_eval)
        for i, il_list in enumerate(illegal_list_batch):
            for j, item in enumerate(il_list):
                if item == 1: #if illegal
                    q_target[i][j] = 0

        

        idx = np.arange(self.batch_size)
        # Recalculate the q-value of the action taken in each state

        q_target[idx, action_batch] = reward_batch + \
        self.gamma*np.max(q_next, axis=1)

     

       

        # Calls training
        # Basic Training: gives input and desired output.
        self.q_eval.dqn.train_on_batch(
            x=[global_state_batch, local_state_batch], y=q_target)
        

        # reduces Epsilon: Network relies less on exploration over time
        if self.counter > self.mem_size and self.epsilon > 0:
            if self.epsilon > 0.05:
                self.epsilon -= 1e-5  # go constant at 25000 steps
            elif self.epsilon <= 0.05:
                self.epsilon = 0.05

    def rare_Exploration(self):
        """
        rare_Exploration Forced exploration if the ai is exploiting. Is an experiment

        :return: If the ai should explore
        :rtype: bool
        """
        # Is used when exploration is zero
        # If the ai is too much exploiting -> Force an exploration
        if self.epsilon >= 0:
            return False

        variance = 0
        container = []
        for i in range(0, self.recent_mem):
            if self.recent_actions[i] not in container:
                container.append(self.recent_actions[i])
                variance += 1
        if variance < self.recent_mem/2:
            return True
        return False

    def save_models(self):
        """
        save_models Save the Networks
        """
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        """
        load_models Load the Networks
        """
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def update_graph(self):
        """
        update_graph Update the q_next Network. Set it to the weights of the q_eval network.
        """
        print("...Updating Network...")
        self.q_next.dqn.set_weights(self.q_eval.dqn.get_weights())
