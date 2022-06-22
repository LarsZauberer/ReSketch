import os
import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, Input, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model, clone_model
import numpy as np
import time

# loss function: Optimizer tries to minimize value.
# Q_value is output by network, q_target is optimal Q_value. Means output of NN should approach q_target
# outside of class to avoid "self" argument


class DeepQNetwork(object):
    def __init__(self, lr, n_actions: int, batch_size: int, name: str,
                 global_input_dims: int, local_input_dims: int, fc1_dims: int = 512, chkpt_dir='tmp/dqn'):
        self.lr = lr  # The optimization learning rate of the network model
        self.n_actions = n_actions  # How many actions the agent has available -> Index of the action to execute
        self.name = name  # The identification name of the agent

        # Inputs
        self.global_input_dims = global_input_dims  # The dimensions of the global input. So the dimensions of the full drawing canvas.
        self.local_input_dims = local_input_dims  # The dimensions of the concentrated input. The local patch of the canvas.
        self.q_target = np.zeros(self.n_actions)  # ? I don't know
        self.fc1_dims = fc1_dims  # Dimensions of the last dense layer
        self.batch_size = batch_size  # ? I don't know -> Something with the agent training

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
        glob_conv1 = Conv2D(32, (8, 8), strides=2,  activation="relu", input_shape=self.global_input_dims,
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

        self.dqn = Model(inputs=[glob_in, loc_in], outputs=[out])

        # Network is ready for calling / Training
        self.dqn.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.lr), metrics=["accuracy"])

        # training: dqn.fit(x=[global_input, local_input], y=[q_target], batch_size=self.batch_size, epochs=self.n_epochs, callbacks=[early_stopping, checkpoint])
        # calling: dqn.predict([global_input_batch, local_input_batch]) or dqn([global_input_batch, local_input_batch])

    def load_checkpoint(self):
        """
        load_checkpoint Load a network checkpoint from the file
        """
        print("...Loading checkpoint...")
        self.dqn = load_model(self.checkpoint_file)

    def save_checkpoint(self):
        """
        save_checkpoint Save a network checkpoint to the file
        """
        print("...Saving checkpoint...")
        self.dqn.save(self.checkpoint_file)


class Agent(object):
    def __init__(self, alpha, gamma, mem_size, epsilon, global_input_dims, local_input_dims, batch_size,
                 replace_target=10000,
                 q_next_dir='paper rebuild stable/nn_memory/q_next', q_eval_dir='paper rebuild stable/nn_memory/q_eval'):

        self.n_actions = local_input_dims[0]*(local_input_dims[1]**2)  # How many action options the agent has. -> Index of the action to choose
        self.action_space = [i for i in range(self.n_actions)]  # All the actions the agent can choose
        self.gamma = gamma  # ? I don't know
        self.mem_size = mem_size  # The allocated memory size
        self.counter = 0  # ? The episode counter?
        self.epsilon = epsilon  # The epsilon value of the agent. The exploration value
        self.batch_size = batch_size  # ? I don't know
        self.replace_target = replace_target  # ? I don't know
        self.global_input_dims = global_input_dims  # The input dimensions of the whole canvas.
        self.local_input_dims = local_input_dims  # The dimensions of the concentrated patch of the canvas

        # ? Why does this seperation exist
        # ? The definitions are maybe not correct
        self.q_next = DeepQNetwork(alpha, self.n_actions, self.batch_size, global_input_dims=global_input_dims,
                                   local_input_dims=local_input_dims, name='q_next', chkpt_dir=q_next_dir)  # The QNetwork to calculate the next value to choose the ai
        self.q_eval = DeepQNetwork(alpha, self.n_actions, self.batch_size, global_input_dims=global_input_dims,
                                   local_input_dims=local_input_dims, name='q_eval', chkpt_dir=q_eval_dir) # The QNetwork to evaluate the action of the agent.

        # Constant helper variables
        glob_mem_shape = (
            self.mem_size, global_input_dims[0], global_input_dims[1], global_input_dims[2])
        loc_mem_shape = (
            self.mem_size, local_input_dims[0], local_input_dims[1], local_input_dims[2])
        
        # Replay buffer
        # ? Maybe some explanation on how the matrizes work
        self.global_state_memory = np.zeros(glob_mem_shape)
        self.local_state_memory = np.zeros(loc_mem_shape)
        self.new_global_state_memory = np.zeros(glob_mem_shape)
        self.new_local_state_memory = np.zeros(loc_mem_shape)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)

        self.recent_mem = 6
        self.recent_actions = np.zeros(self.recent_mem)

    def store_transition(self, global_state: np.array, local_state: np.array, next_gloabal_state: np.array, next_local_state: np.array, action: np.array, reward: np.array):
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
        :param action: The actions the agent is allowed to take
        :type action: np.array
        :param reward: The reward the agent got
        :type reward: np.array
        """
        # ? action i'm not sure
        index = self.counter % self.mem_size

        self.global_state_memory[index] = global_state
        self.local_state_memory[index] = local_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_global_state_memory[index] = next_gloabal_state
        self.new_local_state_memory[index] = next_local_state

    def choose_action(self, global_state, local_state):
        """
        choose_action Agent should choose an action from the action_space

        :param global_state: The whole canvas
        :type global_state: np.array
        :param local_state: The small patch of the canvas
        :type local_state: np.array
        :return: Index of the action the agent want's to take
        :rtype: int
        """
        # Check if the agent should explore
        rand = np.random.random()
        if rand < self.epsilon or self.rare_Exploration():
            action = np.random.choice(self.action_space)
        else:
            # create batch of state (prediciton must be in batches)
            # ? What?
            glob_batch = np.array([global_state])
            loc_batch = np.array([local_state])
            for _ in range(self.batch_size-1):
                glob_batch = np.append(glob_batch, np.array(
                    [np.zeros(self.global_input_dims)]), axis=0)
                loc_batch = np.append(loc_batch, np.array(
                    [np.zeros(self.local_input_dims)]), axis=0)

            # Ask the QNetwork for an action
            actions = self.q_eval.dqn([glob_batch, loc_batch])[0]
            # actions = self.q_eval.dqn.predict([glob_batch, loc_batch], batch_size=self.batch_size)[0]
            # ? What?
            action = int(np.argmax(actions))

        # Save the action to the replay buffer
        # ? Why save it again in the recent_actions
        # ? Wird es nicht nachher noch durch die store_transition function schon gespeichert
        action_ind = self.counter % self.recent_mem
        self.recent_actions[action_ind] = action

        return action

    def learn(self):
        """
        learn The agent/network should learn from the episode
        """
        # update q_next after certain step
        if self.counter % self.replace_target == 0:
            self.update_graph()

        # randomly samples Memory.
        # chooses as many states from Memory as batch_size requires
        max_mem = self.counter if self.counter < self.mem_size else self.mem_size
        batch = np.random.choice(max_mem, self.batch_size)

        # load sampled memory
        global_state_batch = self.global_state_memory[batch]
        local_state_batch = self.local_state_memory[batch]
        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        new_global_state_batch = self.new_global_state_memory[batch]
        new_local_state_batch = self.new_local_state_memory[batch]

        # runs network. also delivers output for training
        q_eval = self.q_eval.dqn([global_state_batch, local_state_batch])
        q_next = self.q_next.dqn(
            [new_global_state_batch, new_local_state_batch])

        # q_eval = self.q_eval.dqn.predict([global_state_batch, local_state_batch], batch_size=self.batch_size)  #target network
        # q_next = self.q_next.dqn.predict([new_global_state_batch, new_local_state_batch], batch_size=self.batch_size) #evaluation network

        # Calculates optimal output for training. ( Bellman Equation !! )
        q_target = np.copy(q_eval)
        idx = np.arange(self.batch_size)
        q_target[idx, action_batch] = reward_batch + \
            self.gamma*np.max(q_next, axis=1)

        # Calls training
        # Basic Training: gives input and desired output.
        self.q_eval.dqn.train_on_batch(
            x=[global_state_batch, local_state_batch], y=q_target)
        # self.q_eval.dqn.fit(x=[global_state_batch, local_state_batch], y=q_target, batch_size=self.batch_size, epochs=10, verbose=0)

        # reduces Epsilon: Network relies less on exploration over time
        if self.counter > self.mem_size and self.epsilon != 0:
            if self.epsilon > 0.05:
                self.epsilon -= 1e-5  # go constant at 25000 steps
            elif self.epsilon <= 0.05:
                self.epsilon = 0.05

    def rare_Exploration(self):
        # ? Not sure
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
