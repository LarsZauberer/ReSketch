import os
import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, Input, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model, clone_model
import numpy as np

# loss function: Optimizer tries to minimize value. 
# Q_value is output by network, q_target is optimal Q_value. Means output of NN should approach q_target
#outside of class to avoid "self" argument
def loss(y_true, y_pred):
            squared_difference = tf.square(y_true - y_pred)
            return tf.reduce_mean(squared_difference, axis=-1)

class DeepQNetwork(object):
    def __init__(self, lr, n_actions, batch_size, name, 
                 global_input_dims, local_input_dims, fc1_dims=512, chkpt_dir='tmp/dqn'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        #inputs
        self.global_input_dims = global_input_dims
        self.local_input_dims = local_input_dims
        self.q_target = np.zeros(self.n_actions)
        self.fc1_dims = fc1_dims
        self.batch_size = batch_size
        #saving / memory
        self.checkpoint_file = os.path.join(chkpt_dir,'deepqnet.ckpt')
        self.params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
        self.build_network()

        
    def build_network(self):
        #global convolution
        glob_in = Input(shape=self.global_input_dims, batch_size=self.batch_size, name="global_input")
        glob_conv1 = Conv2D(32, (8,8), strides=2,  activation="relu", input_shape=self.global_input_dims, padding="same", name="glob_conv1", data_format='channels_first')(glob_in)
        glob_conv2 = Conv2D(64, (4,4), strides=2, activation="relu", name="glob_conv2", padding="same", data_format='channels_first')(glob_conv1)
        glob_conv3 = Conv2D(64, (3,3), strides=1, activation="relu", name="glob_conv3", padding="same", data_format='channels_first')(glob_conv2)
        
        #local convolution
        loc_in = Input(shape=self.local_input_dims, name="local_input", batch_size=self.batch_size) 
        loc_conv1 = Conv2D(128, (11,11), strides=1, activation="relu", name="loc_conv1", padding="same", data_format='channels_first')(loc_in)
        
        #concat
        concat_model = concatenate([glob_conv3, loc_conv1], axis=1)

        #Fully connected Layers
        concat_model = Flatten(name="Flatten", data_format="channels_first")(concat_model)
        dense1 = Dense(self.fc1_dims, activation="relu", name="dense1")(concat_model)
        out = Dense(self.n_actions, activation="relu", name="output")(dense1)

        self.dqn = Model(inputs=[glob_in, loc_in], outputs=[out])
        
        
        

        #Network is ready for calling / Training
        self.dqn.compile(loss=loss, optimizer="adam", metrics=["accuracy"])

        #training: dqn.fit(x=[global_input, local_input], y=[q_target], batch_size=self.batch_size, epochs=self.n_epochs, callbacks=[early_stopping, checkpoint])
        #calling: dqn.predict([global_input_batch, local_input_batch]) or dqn([global_input_batch, local_input_batch])

    
        
    def load_checkpoint(self):
        print("...Loading checkpoint...")
        self.dqn = load_model(self.checkpoint_file, custom_objects={"loss": loss})
        
    def save_checkpoint(self):
        print("...Saving checkpoint...")
        self.dqn.save(self.checkpoint_file)



class Agent(object):
    def __init__(self, alpha, gamma, mem_size, epsilon, global_input_dims, local_input_dims, batch_size,
                 replace_target=10000, 
                 q_next_dir='paper rebuild tf2/nn_memory/q_next', q_eval_dir='paper rebuild tf2/nn_memory/q_eval'):

        self.n_actions = local_input_dims[0]*(local_input_dims[1]**2)
        self.action_space = [i for i in range(self.n_actions)]
        self.gamma = gamma
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replace_target = replace_target
        self.global_input_dims = global_input_dims
        self.local_input_dims = local_input_dims

        self.q_next = DeepQNetwork(alpha, self.n_actions, self.batch_size, global_input_dims=global_input_dims,
                                    local_input_dims=local_input_dims, name='q_next', chkpt_dir=q_next_dir)
        self.q_eval = DeepQNetwork(alpha, self.n_actions, self.batch_size, global_input_dims=global_input_dims,
                                    local_input_dims=local_input_dims, name='q_eval', chkpt_dir=q_eval_dir)

        glob_mem_shape = (self.mem_size, global_input_dims[0], global_input_dims[1], global_input_dims[2])
        loc_mem_shape = (self.mem_size, local_input_dims[0], local_input_dims[1], local_input_dims[2])
        self.global_state_memory = np.zeros(glob_mem_shape)
        self.local_state_memory = np.zeros(loc_mem_shape)
        self.new_global_state_memory = np.zeros(glob_mem_shape)
        self.new_local_state_memory = np.zeros(loc_mem_shape)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)

        self.end = 0

    def store_transition(self, global_state, local_state, next_gloabal_state, next_local_state, action, reward):
        index = self.mem_cntr % self.mem_size
        
        self.global_state_memory[index] = global_state
        self.local_state_memory[index] = local_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_global_state_memory[index] = next_gloabal_state
        self.new_local_state_memory[index] = next_local_state

        self.mem_cntr += 1

    def choose_action(self, global_state, local_state):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            #create batch of state (prediciton must be in batches)
            glob_batch = np.array([global_state])
            loc_batch = np.array([local_state])

            for _ in range(self.batch_size-1):
                glob_batch = np.append(glob_batch, np.array([np.zeros(self.global_input_dims)]), axis=0)
                loc_batch = np.append(loc_batch, np.array([np.zeros(self.local_input_dims)]), axis=0)

            actions = self.q_eval.dqn.predict([glob_batch, loc_batch], batch_size=self.batch_size)[0]
            action = int(np.argmax(actions))
        
        return action

    def learn(self):
        #update q_next after certain step
        if self.mem_cntr % self.replace_target == 0:
            self.update_graph()

        #randomly samples Memory.
        #chooses as many states from Memory as batch_size requires 
        max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size
        batch = np.random.choice(max_mem, self.batch_size)
        
        #load sampled memory
        global_state_batch = self.global_state_memory[batch]
        local_state_batch = self.local_state_memory[batch]
        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        new_global_state_batch = self.new_global_state_memory[batch]
        new_local_state_batch = self.new_local_state_memory[batch]

        #runs network. also delivers output for training
        q_eval = self.q_eval.dqn.predict([global_state_batch, local_state_batch], batch_size=self.batch_size)  #target network
        q_next = self.q_next.dqn.predict([new_global_state_batch, new_local_state_batch], batch_size=self.batch_size) #evaluation network
        
        #Calculates optimal output for training. ( Bellman Equation !! )
        q_target = np.copy(q_eval)
        idx = np.arange(self.batch_size)
        q_target[idx, action_batch] = reward_batch + self.gamma*np.max(q_next, axis=1)

        #Calls training
        #Basic Training: gives input and desired output.
        self.q_eval.dqn.fit(x=[global_state_batch, local_state_batch], y=q_target, batch_size=self.batch_size, epochs=10, verbose=0)

        #reduces Epsilon: Network relies less on exploration over time
        if self.mem_cntr > self.mem_size:
            if self.epsilon > 0.05:
                self.epsilon -= 8e-5 #go constant at 50000 frames
            elif self.epsilon <= 0.1:
                self.epsilon = 0.05
        


    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def update_graph(self):
        del self.q_next
        self.q_next = clone_model(self.q_eval) 