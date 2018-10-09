import tensorflow as tf
import numpy as np
import tensorflow.contrib as tc

import matplotlib.pyplot as plt


class NN():
    def __init__(self):
        self.layer_norm = True
        self.e_NN = 1.0
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        loc = './cartpole/RodrigoPerez_network'
        with self.graph.as_default():
            self.build_neural_network_model()
            self.init = tf.global_variables_initializer()  # initialize the graph
            self.sess = tf.Session()
            self.sess.run(self.init)
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, loc)
            self.policy = self.graph.get_operation_by_name('action').outputs[0]
        self.observation_list = np.array([])
        self.y__list = np.array([])
#        self.merged_summary = tf.summary.merge_all()
        # self.writer = tf.summary.FileWriter("/home/rodrigo/Documents/Tesis/COACH CartPole Python/1")
        # self.writer.add_graph(self.sess.graph)
        self.buffer_train_cnt = 0  # buffer train counter
        # buffer parameters
        self.use_buffer = False
        self.buffer_size = 100
        self.buffer_sampling_size = 30

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
        # self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1])) # cross entropy
        self.loss = 0.5 * tf.reduce_mean(tf.square(self.y - self.y_))
        # define training step
        self.train_step = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.0).minimize(self.loss)

        #tf.summary.histogram("hidden weights", self.W_fc1)
        #tf.summary.histogram("activation", self.h_fc1)

    def action(self, observation):
        observation = np.array(observation).reshape(1, 4)
        #action = self.y.eval(session=self.sess, feed_dict={self.input: observation})
        action = self.sess.run(self.policy, feed_dict={self.input: observation})
        if action[0] > 1:
            action[0] = 1
        if action[0] < -1:
            action[0] = -1
        return action[0]

    def get_feedback_signal(self, observation, agent_output, episode):
        action = self.action(observation)
        diff = action - agent_output
        abs_diff = np.abs(diff)
        # feedback_prob = abs_diff[h_max]
        egreedy = True
        if egreedy:
            feedback_prob = 0.6*np.exp(-0.0003*episode)  # 0.015
        self.h = np.array([0])
        error_prob = 0  # 0.07 -> 20%; 0.035 -> 10%

        if np.random.uniform() < feedback_prob:
            self.h = np.sign(diff)
            # Give erroneous feedback
            #if np.random.uniform() < error_prob:
             #   self.h = self.h * -1
        return self.h
