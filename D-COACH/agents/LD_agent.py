import tensorflow as tf
import numpy as np
from agents.agent_base import AgentBase
from models import fully_connected_layers


class Agent(AgentBase):
    def __init__(self, load_policy=False, learning_rate=0.001, dim_a=3, dim_state=4,
                 fc_layers_neurons=100, loss_function_type='mean_squared', policy_loc='./racing_car_m2/network',
                 action_upper_limits='1,1', action_lower_limits='-1,-1', e='1'):

        super(Agent, self).__init__(dim_a=dim_a, policy_loc=policy_loc, action_upper_limits=action_upper_limits,
                                    action_lower_limits=action_lower_limits, e=e, load_policy=load_policy,
                                    loss_function_type=loss_function_type, learning_rate=learning_rate,
                                    fc_layers_neurons=fc_layers_neurons, low_dim_input_shape=dim_state)

        self.low_dim_input_shape = dim_state

    def _build_network(self, dim_a, params):
        with tf.variable_scope('base'):
            # Input data
            x = tf.placeholder(tf.float32, [None, params['low_dim_input_shape']], name='input')

            # Build fully connected layers
            self.y, loss = fully_connected_layers(x, dim_a,
                                                  params['fc_layers_neurons'],
                                                  params['loss_function_type'])

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'base')
        self.train_step = tf.train.MomentumOptimizer(learning_rate=params['learning_rate'],
                                                     momentum=0.00).minimize(loss, var_list=variables)

        # Initialize tensorflow
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver()

    def _preprocess_observation(self, observation):
        self.low_dim_observation = np.reshape(observation, [-1, self.low_dim_input_shape])

