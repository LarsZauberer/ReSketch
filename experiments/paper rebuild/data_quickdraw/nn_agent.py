import os
import tensorflow as tf
import numpy as np

class DeepQNetwork(object):
    def __init__(self, lr, n_actions, name, fc1_dims=1024,
                global_input_dims=(84,84,4), local_input_dims=(11,11,2), chkpt_dir='tmp/dqn'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name

        self.fc1_dims = fc1_dims
        self.global_input_dims = global_input_dims
        self.local_input_dims = local_input_dims

        self.sess = tf.Session()
        self.build_network()
        self.sess.run(tf.global_variables_initializer())

        #saving / memory
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir,'deepqnet.ckpt')
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def build_network(self):
        with tf.variable_scope(self.name):
            self.global_input = tf.placeholder(tf.float32, shape=[None, *self.global_input_dims], name='global inputs')
            self.local_input = tf.placeholder(tf.float32, shape=[None, *self.local_input_dims], name='local inputs')
            self.q_target = tf.placeholder(tf.float32, shape=[None, self.n_actions], name='q_value')

            #global convolution
            glob_conv1 = tf.layers.conv2d(inputs=self.global_input, filters=32, kernel_size=(8,8), strides=4, name='glob_conv1', 
                                    kernel_initializer=tf.variance_scaling_initializer(scale=2))
            glob_conv1_activated = tf.nn.relu(glob_conv1)

            glob_conv2 = tf.layers.conv2d(inputs=glob_conv1_activated, filters=64, kernel_size=(4,4), strides=2, name='glob_conv2',
                                    kernel_initializer=tf.variance_scaling_initializer(scale=2))
            glob_conv2_activated = tf.nn.relu(glob_conv2)

            glob_conv3 = tf.layers.conv2d(inputs=glob_conv2_activated, filters=64, kernel_size=(3,3),strides=1, name='glob_conv3',
                                    kernel_initializer=tf.variance_scaling_initializer(scale=2))
            glob_conv3_activated = tf.nn.relu(glob_conv3)

            #local convolution
            loc_conv1 = tf.layers.conv2(inputs=self.local_input, filters=128, kernel_size=(11,11), strides=1, name="loc_conv1",
                                    kernel_initializer=tf.variance_scaling_initializer(scale=2))
            loc_conv1_activated = tf.nn.relu(loc_conv1)

            #concat global and local stream
            fcl_input = tf.concat([glob_conv3_activated, loc_conv1_activated], -1)

            dense1 = tf.layers.dense(fcl_input, units=self.fc1_dims, activation=tf.nn.relu,
                                    kernel_initializer=tf.variance_scaling_initializer(scale=2))

            #This Function runs NN. it is the output layer and recursively calls previous layers.
            self.Q_values = tf.layers.dense(dense1, units=self.n_actions,
                                    kernel_initializer=tf.variance_scaling_initializer(scale=2))

            # loss function: Optimizer tries to minimize value. 
            # Q_value is output by network, q_target is optimal Q_value 
            # Means output of NN should approach q_target
            self.loss = tf.reduce_mean(tf.square(self.Q_values - self.q_target))

            # performs training. minimizes loss function (Adam)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def load_checkpoint(self):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("...Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)


class Agent(object):
    def __init__(self, alpha, gamma, mem_size, n_actions, epsilon, batch_size,
                 replace_target=10000, global_input_dims=(4,84,84), local_input_dims=(2,11,11),
                 q_next_dir='nn_memory/q_next', q_eval_dir='nn_memory/q_eval'):

        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replace_target = replace_target

        self.q_next = DeepQNetwork(alpha, n_actions, global_input_dims=global_input_dims,
                                    local_input_dims=local_input_dims, name='q_next', chkpt_dir=q_next_dir)
        self.q_eval = DeepQNetwork(alpha, n_actions, global_input_dims=global_input_dims,
                                    local_input_dims=local_input_dims, name='q_eval', chkpt_dir=q_eval_dir)

        self.global_state_memory = np.zeros((self.mem_size, global_input_dims))
        self.local_state_memory = np.zeros((self.mem_size, local_input_dims))
        self.new_global_state_memory = np.zeros((self.mem_size, global_input_dims))
        self.new_local_state_memory = np.zeros((self.mem_size, local_input_dims))

        self.action_memory = np.zeros((self.mem_size, self.n_actions), dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)

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
            actions = self.q_eval.sess.run(self.q_eval.Q_values,  
                feed_dict={self.q_eval.global_input: global_state, self.q_eval.local_input: local_state})
            action = np.argmax(actions)
        return action

    def learn(self):
        #update q_next after certain step
        if self.mem_cntr % self.replace_target == 0:
            self.update_graph()

        #randomly samples Memory. 
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
        #target network
        q_eval = self.q_eval.sess.run(self.q_eval.Q_values, 
            feed_dict={
                self.q_eval.global_input: global_state_batch, 
                self.q_eval.local_input: local_state_batch
            })
        #evaluation network
        q_next = self.q_next.sess.run(self.q_next.Q_values,
            feed_dict={
                self.q_next.global_input: new_global_state_batch, 
                self.q_next.local_input: new_local_state_batch 
            })

        #Calculates optimal output for training. ( Bellman Equation !! )
        q_target = q_eval.copy()
        idx = np.arange(self.batch_size)
        q_target[idx, action_batch] = reward_batch + self.gamma*np.max(q_next, axis=1)

        #q_target = np.zeros(self.batch_size)
        #q_target = reward_batch + self.gamma*np.max(q_next, axis=1)*terminal_batch

        #Calls training
        #Basic Training: gives input and desired output.
        _ = self.q_eval.sess.run(self.q_eval.train_op,
            feed_dict={
                self.q_eval.local_input: local_state_batch, 
                self.q_eval.global_input: global_state_batch,
                self.q_eval.q_target: q_target
            })

        #reduces Epsilon: Network relies less on exploration over time
        if self.mem_cntr > 25000:#200000:
            if self.epsilon > 0.05:
                self.epsilon -= 4e-7
            elif self.epsilon <= 0.05:
                self.epsilon = 0.05


    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def update_graph(self):
        t_params = self.q_next.params
        e_params = self.q_eval.params
        for t, e in zip(t_params, e_params):
            #copy from q_eval to q_next
            self.q_eval.sess.run(tf.assign(t,e))