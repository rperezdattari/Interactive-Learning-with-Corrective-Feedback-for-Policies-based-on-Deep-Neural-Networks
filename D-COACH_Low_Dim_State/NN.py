import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np


class NN():
    def __init__(self, buffer):
        self.e_NN = 1.0
        self.layer_norm = True
        self.build_neural_network_model()
        self.init = tf.global_variables_initializer() # initialize the graph
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.saver = tf.train.Saver()
        #self.saver.restore(self.sess, './cartpole/RodrigoPerez_network')
        self.observation_list = []
        self.y__list = []
        self.first = True
        self.merged_summary = tf.summary.merge_all()
        #self.writer = tf.summary.FileWriter("/home/rodrigo/Documents/Tesis/COACH CartPole Python/1")
        #self.writer.add_graph(self.sess.graph)
        # buffer parameters
        self.use_buffer = buffer
        self.buffer_size = 1000
        self.buffer_sampling_size = 100
        self.automatic_buffer_train = True


    def build_neural_network_model(self):
        # correct labels
        n_neurons = 64
        self.y_ = tf.placeholder(tf.float32, [None, 1])

        # input data
        self.input = tf.placeholder(tf.float32, [None, 4])

        self.x = tf.layers.dense(self.input, 64)
        if self.layer_norm:
            self.x = tc.layers.layer_norm(self.x, center=True, scale=True)
        self.x = tf.nn.relu(self.x)

        self.x = tf.layers.dense(self.x, 64)
        if self.layer_norm:
            self.x = tc.layers.layer_norm(self.x, center=True, scale=True)
        self.x = tf.nn.relu(self.x)

        self.x = tf.layers.dense(self.x, 1,
                                 kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        self.y = tf.nn.tanh(self.x, name='action')

        # define the loss function
        self.error = tf.placeholder(tf.float32, [None, 1])
        self.y_ = tf.placeholder(tf.float32, [None, 1])
        # define the loss function
        #self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1])) # cross entropy
        self.loss = 0.5 * tf.reduce_mean(tf.square(self.y - self.y_))
        # define training step
        self.train_step = tf.train.MomentumOptimizer(learning_rate=0.0003, momentum=0.0).minimize(self.loss)

        #tf.summary.histogram("hidden weights", self.W_fc1)
        #tf.summary.histogram("activation", self.h_fc1)

    # Train NN model
    def update(self, h, observation):
        observation = np.array(observation).reshape(1, 4)
        error = np.array(h * self.e_NN).reshape(1, 1)
        action = self.y.eval(session=self.sess, feed_dict={self.input: observation})

        if action + h * self.e_NN > 1:
            y_ = np.array([1]).reshape(1, 1)
        elif action + h * self.e_NN < -1:
            y_ = np.array([-1]).reshape(1, 1)
        else:
            y_ = np.array(action).reshape(1, 1) + error

        #lastW_fc1 = self.sess.run(self.W_fc1)
        #s = self.sess.run(self.merged_summary, feed_dict={self.input: observation, self.y_: y_})
        #self.writer.add_summary(s, self.i)
        self.sess.run(self.train_step, feed_dict={self.input: observation, self.y_: y_})
        if self.use_buffer:
            self.train_NN_model_from_buffer()  # Update weights from buffer each time feedback is received
            self.append_training_pairs(observation, y_)  # For buffer


    def action(self, observation):
        observation = np.array(observation).reshape(1, 4)
        action = self.y.eval(session=self.sess, feed_dict={self.input: observation})
        if action[0] > 1:
            action[0] = 1
        if action[0] < -1:
            action[0] = -1
        return action[0]

    def append_training_pairs(self, observation, y):
        self.observation_list.append(observation)
        self.y__list.append(y)

    def train_NN_model_from_buffer(self):
        if self.use_buffer and len(self.y__list) > 50 and self.automatic_buffer_train:
            # Generate uniform random sample from data
            if len(self.y__list) < self.buffer_sampling_size:
                n_samples = int(len(self.y__list)/4) + 1
            else:
                n_samples = self.buffer_sampling_size
            if len(self.y__list) > self.buffer_size:
                self.observation_list = self.observation_list[-self.buffer_size:]
                self.y__list = self.y__list[-self.buffer_size:]
            sampled_observation, sampled_y__list = self.random_sampling(self.observation_list, self.y__list, n_samples)
            # Train
            #print('Training from buffer...', n_samples)
            self.sess.run(self.train_step, feed_dict={self.input: sampled_observation, self.y_: sampled_y__list})

    def random_sampling(self, observation_list, y__list, n_samples):
        selected = np.random.choice(len(observation_list), n_samples, replace=False)
        sampled_observation_list = []
        sampled_y__list = []
        for i in selected:
            sampled_observation_list.append(observation_list[i])
            sampled_y__list.append(y__list[i])

        return np.reshape(sampled_observation_list, [-1, 4]), np.reshape(sampled_y__list, [-1, 1])

    def buffer_auto_training(self, last_episode_received_feedback):
        if last_episode_received_feedback > 0.0:
            self.automatic_buffer_train = True
        else:
            self.automatic_buffer_train = False
            self.observation_list = []
            self.y__list = []
        print('automatic buffer update state:', self.automatic_buffer_train)

    def new_episode(self):
        self.renew_buffer_cnt = 0

    def save_params(self, name):
        save_path = self.saver.save(self.sess, './cartpole/' + name + '_network')
