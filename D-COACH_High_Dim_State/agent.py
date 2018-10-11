import tensorflow as tf
import numpy as np
from autoencoder import TrainAE, AE
from models import fully_connected_layers
from tools.functions import str_2_array, observation_to_gray
import os


class Agent:
    def __init__(self, train_ae=True, load_policy=False, learning_rate=0.001,
                 dim_a=3, loss_function_type='mean_squared', policy_loc='./racing_car_m2/network',
                 image_size=64, action_upper_limits='1,1', action_lower_limits='-1,-1', e='1',
                 ae_loc='graphs/autoencoder/CarRacing-v0/conv_layers_64x64'):
        # Initialize variables
        self.observation = None
        self.y_label = None
        self.e = np.array(str_2_array(e, type_n='float'))
        self.dim_a = dim_a

        self.policy_loc = policy_loc + 'network'

        self.action_upper_limits = str_2_array(action_upper_limits)
        self.action_lower_limits = str_2_array(action_lower_limits)

        self._build_network(train_ae, ae_loc, dim_a, loss_function_type, learning_rate, load_policy)

        if load_policy:
            self._load_network()

        self.image_size = image_size

    def _build_network(self, train_ae, ae_loc, dim_a, loss_function_type, learning_rate, load_policy):
        with tf.variable_scope('base'):
            # Initialize graph
            if train_ae:
                ae_trainer = TrainAE()
                ae_trainer.run(train=True, show_performance=True)

            self.AE = AE(ae_loc=ae_loc)
            ae_encoder = self.AE.latent_space
            self.ae_encoder_shape = ae_encoder.get_shape()
            self.low_dim_input = tf.placeholder(tf.float32, [None, self.ae_encoder_shape[1],
                                                             self.ae_encoder_shape[2],
                                                             self.ae_encoder_shape[3]],
                                                name='input')

            self.low_dim_input = tf.identity(self.low_dim_input, name='low_dim_input')

            # build fully connected layers
            self.y, loss = fully_connected_layers(self.low_dim_input, dim_a, loss_function_type)

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'base')
        self.train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                     momentum=0.00).minimize(loss, var_list=variables)

        # initialize tensorflow
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver()

    def _load_network(self):
            self.saver.restore(self.sess, self.policy_loc)

    def update(self, h, observation):
        observation = observation_to_gray(observation, self.image_size)

        self.observation = self.AE.conv_representation(observation)

        action = self.y.eval(session=self.sess, feed_dict={'base/input:0': self.observation})

        error = np.array(h * self.e).reshape(1, self.dim_a)
        self.y_label = []

        for i in range(self.dim_a):
            self.y_label.append(np.clip(action[0, i] + error[0, i],
                                        self.action_lower_limits[i], self.action_upper_limits[i]))

        self.y_label = np.array(self.y_label).reshape(1, self.dim_a)

        self.sess.run(self.train_step, feed_dict={'base/input:0': self.observation,
                                                  'base/label:0': self.y_label})

    def batch_update(self, batch):
        state_batch = [np.array(pair[0]) for pair in batch]
        ylabel_batch = [np.array(pair[1]) for pair in batch]

        self.sess.run(self.train_step, feed_dict={'base/input:0': state_batch,
                                                  'base/label:0': ylabel_batch})

    def action(self, observation):
        observation = observation_to_gray(observation, self.image_size)

        observation = self.AE.conv_representation(observation)

        action = self.y.eval(session=self.sess, feed_dict={'base/input:0': observation})
        out_action = []

        for i in range(self.dim_a):
            action[0, i] = np.clip(action[0, i], self.action_lower_limits[i], self.action_upper_limits[i])
            out_action.append(action[0, i])

        return np.array(out_action)

    def ae_output(self, observation):
        observation = observation_to_gray(observation, self.image_size)
        return self.AE.output(observation)

    def last_step(self):
        return [self.observation.reshape(self.ae_encoder_shape[1:]), self.y_label.reshape(self.dim_a)]

    def save_params(self):
        if not os.path.exists(self.policy_loc):
            os.makedirs(self.policy_loc)

        self.saver.save(self.sess, self.policy_loc)

    def new_episode(self):
        pass

