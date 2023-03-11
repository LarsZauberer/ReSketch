import os
import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, Dense, Flatten, Lambda, Activation, Input, concatenate
#from tensorflow.keras.utils import plot_model

import numpy as np
from time import sleep
from pathlib import Path
import logging


log = logging.getLogger("nn_agent")


class DeepQNetwork(object):
    def __init__(self, lr, n_actions: int, batch_size: int, name: str,
                 global_input_dims: int, local_input_dims: int, fc1_dims: int = 1024):
        
        self.lr = lr  # The optimization learning rate of the network model
        self.n_actions = n_actions  # How many actions the agent has available -> Index of the action to execute
        self.name = name  # The identification name of the agent

        # Inputs
        self.global_input_dims = global_input_dims  # The dimensions of the global input. So the dimensions of the full drawing canvas.
        self.local_input_dims = local_input_dims  # The dimensions of the concentrated input. The local patch of the canvas.
        self.fc1_dims = fc1_dims  # Dimensions of the last dense layer
        self.batch_size = batch_size  # How many inputs sending into the network

        # Generate the network
        self.build_network()

    def build_network(self):
        """
        build_network Generate a keras DeepQNetwork. The model will be saved in the self.dqn variable.
        """
        # global convolution
        glob_in = Input(shape=self.global_input_dims,
                        batch_size=self.batch_size, name="Global_Stream")
        glob_conv1 = Conv2D(32, (8, 8), strides=2,  activation="relu", input_shape=self.global_input_dims,
                            padding="same", name="Global_Conv_1", data_format='channels_first')(glob_in)
        glob_conv2 = Conv2D(64, (4, 4), strides=2, activation="relu", name="Global_Conv_2",
                            padding="same", data_format='channels_first')(glob_conv1)
        glob_conv3 = Conv2D(64, (3, 3), strides=1, activation="relu", name="Global_Conv_3",
                            padding="same", data_format='channels_first')(glob_conv2)

        # local convolution
        loc_in = Input(shape=self.local_input_dims,
                       name="Local_Stream", batch_size=self.batch_size)
        loc_conv1 = Conv2D(128, (11, 11), strides=1, activation="relu",
                           name="Local_Conv_1", padding="same", data_format='channels_first')(loc_in)

        # concat
        concat_model = concatenate([glob_conv3, loc_conv1], name="Concatenation", axis=1)

        # Fully connected Layers
        concat_model = Flatten(
            name="Flatten", data_format="channels_first")(concat_model)
        dense1 = Dense(self.fc1_dims, activation="relu",
                       name="Fully_Connected_1")(concat_model)

    
        out = Dense(self.n_actions, name="Action-Space")(dense1)

        """ if self.softmax:
            softmax_temp = Lambda(lambda x: x / self.softmax_temp, name="Softmax_Temperature")(out)
            softmax = Activation("softmax", name="Softmax")(softmax_temp)
            self.dqn = Model(inputs=[glob_in, loc_in], outputs=[softmax])
        else: """

        # Inputs of the network are global and local states: glob_in = [4x28x28], loc_in = [2x7x7]
        # Output of the netword are Q-values. each Q-value represents an action
        self.dqn = Model(inputs=[glob_in, loc_in], outputs=[out])

        # Network is ready for calling / Training
        self.dqn.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.lr), metrics=["accuracy"])

        #plot_model(self.dqn, to_file=f"src/images/{self.name}.png", show_shapes=True)


        #calling: dqn([global_state_batch, local_state_batch])
        #training: dqn.train_on_batch(x=[global_state_batch, local_state_batch], y=q_target)

    def load_checkpoint(self, path):
        """
        load_checkpoint Load a network checkpoint from the file
        """
        log.info("...Loading checkpoint...")

        path = Path(path + '/deepqnet.ckpt')
        self.dqn.load_weights(path)

    def save_checkpoint(self, path):
        """
        save_checkpoint Save a network checkpoint to the file
        """
        log.info("...Saving checkpoint...")

        path = Path(path + '/deepqnet.ckpt')
        self.dqn.save_weights(path)


    


class Agent(object):
    def __init__(self, alpha, gamma, mem_size, epsilon, epsilon_episodes, global_input_dims, local_input_dims, batch_size, 
                 softmax, softmax_temp : float = 0.05, replace_target=1000):

        self.n_actions = local_input_dims[0]*(local_input_dims[1]**2)  # How many action options the agent has. -> Index of the action to choose
        self.n_actions += 1 # Stop Action
        self.action_space = [i for i in range(self.n_actions)]  # All the actions the agent can choose
        self.gamma = gamma  # Is the learnrate
        self.mem_size = mem_size  # The allocated memory size (The number of slots for saved observation)
        self.counter = 0  # Counter of every step
        self.batch_size = batch_size  # How big each batch of inputs is
        self.replace_target = replace_target  # When to update the q_next network
        self.global_input_dims = global_input_dims  # The input dimensions of the whole canvas.
        self.local_input_dims = local_input_dims  # The dimensions of the concentrated patch of the canvas
        self.softmax = softmax
        self.softmax_temp = softmax_temp

        self.q_next = DeepQNetwork(alpha, self.n_actions, self.batch_size, global_input_dims=global_input_dims,
                                   local_input_dims=local_input_dims, name='q_next')  # The QNetwork to compute the q-values on the next state of the canvas
        self.q_eval = DeepQNetwork(alpha, self.n_actions, self.batch_size, global_input_dims=global_input_dims,
                                   local_input_dims=local_input_dims, name='q_eval') # The QNetwork to compute the q-values on the current state of the canvas

        self.epsilon_episodes = epsilon_episodes
        self.start_epsilon = epsilon
        self.epsilon = self.start_epsilon

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
        if rand < self.epsilon or replay_fill:
            if replay_fill:
                action = np.random.choice([i for i, el in enumerate(illegal_list) if el != 1 and i != 98])
            else:
                action = np.random.choice([i for i, el in enumerate(illegal_list) if el != 1])
        else:    
            # create batch of states (prediciton must be in batches)
            glob_batch = np.array([global_state])
            loc_batch = np.array([local_state])

            
            # Ask the QNetwork for an action
            actions = np.array(self.q_eval.dqn([glob_batch, loc_batch])[0])


            """ for i, el in enumerate(illegal_list):
                if el == 1:
                    actions[i] = 0
            try:
                actions /= np.sum(actions)
            except:
                print(np.sum(actions))
                print(actions) """

            while illegal_list[np.argmax(actions)] == 1:
                actions[np.argmax(actions)] = -1


            # Take the index of the maximal value -> action
            action = int(np.argmax(actions))
            

        return action

        


    def choose_action_softmax(self, global_state: np.array, local_state: np.array, illegal_list : np.array, replay_fill: bool = False):
        if replay_fill:
            action = np.random.choice([i for i, el in enumerate(illegal_list) if el != 1 and i != 98])
        else:
            rand = np.random.random()
            if rand < 0.003:
                return 98

            glob_batch = np.array([global_state])
            loc_batch = np.array([local_state])

            # Ask the QNetwork for an action
            actions = np.array(self.q_eval.dqn([glob_batch, loc_batch])[0])

            for i, el in enumerate(illegal_list):
                if int(el) == 1:
                    actions[i] = 0

            actions = list(enumerate(actions))
            actions = sorted(actions, key=lambda x: x[1])
            actions = [(0, 0) for _ in actions[:-5]] + actions[-5:]
            """ action_sum = np.sum([i[1] for i in actions])
            actions = [(i[0], i[1]/action_sum) for i in actions] """


            
            probabilities = [i[1] for i in actions]

            probabilities = list(probabilities[:-5]) + list(self.apply_softmax(probabilities[-5:]))
            

            action = np.argmax(np.random.multinomial(1000, probabilities))


            #action = np.random.choice(len(actions), 1, p=probabilities)

    
            action = actions[action][0]

            while(illegal_list[action] == 1):
                #action = np.random.choice(len(actions), 1, p=probabilities)

                action = np.argmax(np.random.multinomial(1000, probabilities))
                action = actions[action][0]
                
        return action
        

    
    def apply_softmax(self, actions):

        actions = np.array(actions) / self.softmax_temp
        actions = np.exp(actions)/np.exp(actions).sum()
        actions[-1] += 1 - actions.sum()

        return actions


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

        q_target = np.copy(q_eval)
        for i, il_list in enumerate(illegal_list_batch):
            for j, item in enumerate(il_list):
                if item == 1: #if illegal
                    q_target[i][j] = 0


        # Calculates optimal output for training. ( Bellman Equation !! )
        idx = np.arange(self.batch_size)
        q_target[idx, action_batch] = reward_batch + \
        self.gamma*np.max(q_next, axis=1)

        # Calls training
        # Basic Training: gives input and desired output.
        self.q_eval.dqn.train_on_batch(
            x=[global_state_batch, local_state_batch], y=q_target)
        

    def reduce_epsilon(self):
         # reduces Epsilon: Network relies less on exploration over time
        if self.counter > self.mem_size and self.epsilon > 0:
            if self.epsilon > 0.05:
                epsilon_diff = self.start_epsilon-0.05
                self.epsilon -=  epsilon_diff/self.epsilon_episodes

                self.epsilon -= 1e-5  # go constant at 25000 steps
            else:
                self.epsilon = 0.05

    def save_models(self, path):
        """
        save_models Save the Networks
        """
        self.q_eval.save_checkpoint(path + "/q_eval")
        self.q_next.save_checkpoint(path + "/q_next")

    def load_models(self, path):
        """
        load_models Load the Networks
        """
        self.q_eval.load_checkpoint(path + "/q_eval")
        self.q_next.load_checkpoint(path + "/q_next")

    def update_graph(self):
        """
        update_graph Update the q_next Network. Set it to the weights of the q_eval network.
        """
        #log.info("...Updating Network...")
        self.counter += 1
        if self.counter % self.replace_target == 0 and self.counter > 0:
            self.q_next.dqn.set_weights(self.q_eval.dqn.get_weights())

    def set_softmax_temp(self, temp):
        self.softmax_temp = temp
        self.softmax_temp = temp
