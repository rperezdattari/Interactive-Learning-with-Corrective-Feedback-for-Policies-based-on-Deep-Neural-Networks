import tensorflow as tf
from agents.agent_base import AgentBase
import numpy as np


class Agent(AgentBase):
    def __init__(self, load_policy=False, learning_rate=0.001, dim_a=3, dim_state=4,
                 fc_layers_neurons=100, loss_function_type='mean_squared', policy_loc='./racing_car_m2/network',
                 action_upper_limits='1,1', action_lower_limits='-1,-1', e='1'):

        super(Agent, self).__init__(dim_a=dim_a, policy_loc=policy_loc, action_upper_limits=action_upper_limits,
                                    action_lower_limits=action_lower_limits, e=e, load_policy=load_policy,
                                    loss_function_type=loss_function_type, learning_rate=learning_rate,
                                    fc_layers_neurons=fc_layers_neurons, dim_state=dim_state)

        self.dim_state = dim_state

    def _build_network(self, dim_a, params):
        with tf.variable_scope('base'):
            self.y_ = tf.placeholder(tf.float32, [None, dim_a])

            # input data
            self.input = tf.placeholder(tf.float32, [None, params['dim_state']], name='input')
            self.x = tf.layers.dense(self.input, params['fc_layers_neurons'])
            self.x = tf.nn.relu(self.x)
            self.x = tf.layers.dense(self.x, params['fc_layers_neurons'])
            self.x = tf.nn.relu(self.x)
            self.x = tf.layers.dense(self.x, dim_a,
                                     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

            self.y = tf.nn.tanh(self.x, name='action')
            self.error = tf.placeholder(tf.float32, [None, dim_a])
            self.y_ = tf.placeholder(tf.float32, [None, dim_a], name='label')

            # Define the loss function
            self.loss = 0.5 * tf.reduce_mean(tf.square(self.y - self.y_))

        # define training step
        self.train_step = tf.train.MomentumOptimizer(learning_rate=params['learning_rate'],
                                                     momentum=0.0).minimize(self.loss)

        # Initialize tensorflow
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver()

    def _preprocess_observation(self, observation):
        self.low_dim_observation = np.reshape(observation, [-1, self.dim_state])

