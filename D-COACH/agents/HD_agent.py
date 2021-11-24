import tensorflow as tf
import numpy as np
from autoencoder import TrainAE, AE
from models import fully_connected_layers
from tools.functions import observation_to_gray, FastImagePlot
from agents.agent_base import AgentBase
import cv2
import tf_slim as slim

class Agent(AgentBase):
    def __init__(self, train_ae=True, load_policy=False, learning_rate=0.001,
                 dim_a=3, fc_layers_neurons=100, loss_function_type='mean_squared',
                 policy_loc='./racing_car_m2/network', image_size=64, action_upper_limits='1,1',
                 action_lower_limits='-1,-1', e='1', ae_loc='graphs/autoencoder/CarRacing-v0/conv_layers_64x64',
                 show_ae_output=True, show_state=True, resize_observation=True):

        super(Agent, self).__init__(dim_a=dim_a, policy_loc=policy_loc, action_upper_limits=action_upper_limits,
                                    action_lower_limits=action_lower_limits, e=e, load_policy=load_policy,
                                    train_ae=train_ae, ae_loc=ae_loc, loss_function_type=loss_function_type,
                                    learning_rate=learning_rate, fc_layers_neurons=fc_layers_neurons)

        # High-dimensional state initialization
        self.resize_observation = resize_observation
        self.image_size = image_size
        self.show_state = show_state
        self.show_ae_output = show_ae_output

        if self.show_state:
            self.state_plot = FastImagePlot(1, np.zeros([image_size, image_size]),
                                            image_size, 'Image State', vmax=0.5)

        if self.show_ae_output:
            self.ae_output_plot = FastImagePlot(2, np.zeros([image_size, image_size]),
                                                image_size, 'Autoencoder Output', vmax=0.5)

    def _build_network(self, dim_a, params):
        with tf.compat.v1.variable_scope('base'):
            # Initialize graph
            if params['train_ae']:
                ae_trainer = TrainAE()
                ae_trainer.run(train=True, show_performance=True)

            self.AE = AE(ae_loc=params['ae_loc'])
            ae_encoder = self.AE.latent_space
            self.low_dim_input_shape = ae_encoder.get_shape()[1:]
            # self.low_dim_input_shape = (64,64,1)
            self.low_dim_input = tf.compat.v1.placeholder(tf.float32, [None, self.low_dim_input_shape[0],
                                                             self.low_dim_input_shape[1],
                                                             self.low_dim_input_shape[2]],
                                                name='input')

            self.low_dim_input = tf.identity(self.low_dim_input, name='low_dim_input')

            # Build fully connected layers
            self.y, loss = fully_connected_layers(slim.flatten(self.low_dim_input), dim_a,
                                                  params['fc_layers_neurons'],
                                                  params['loss_function_type'])

        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'base')
        self.train_step = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=params['learning_rate']).minimize(loss, var_list=variables)

        # Initialize tensorflow
        init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session()
        self.sess.run(init)
        self.saver = tf.compat.v1.train.Saver()

    def _preprocess_observation(self, observation):
        if self.resize_observation:
            observation = cv2.resize(observation, (self.image_size, self.image_size))
        self.high_dim_observation = observation_to_gray(observation, self.image_size)
        # self.low_dim_observation = observation_to_gray(observation, self.image_size)
        self.low_dim_observation = self.AE.conv_representation(self.high_dim_observation)  # obtain latent space from AE

    def time_step_info(self, t):
        if t % 4 == 0 and self.show_state:
            self.state_plot.refresh(self.high_dim_observation)

        if (t+2) % 4 == 0 and self.show_ae_output:
            self.ae_output_plot.refresh(self.AE.output(self.high_dim_observation))
