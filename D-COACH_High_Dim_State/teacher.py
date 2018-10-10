import tensorflow as tf
import numpy as np
from tools.functions import observation_to_gray, str_2_array
from autoencoder import AE
import time


class Teacher:
    def __init__(self, network='FNN', method=1, image_size=64, dim_a=3,
                 action_lower_limits='0,0,0', action_upper_limits='1,1,1',
                 loc='graphs/teacher/CarRacing-v0/network', exp='1', error_prob=0):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(loc + '.meta', clear_devices=True)
            self.init = tf.global_variables_initializer()  # initialize the graph
            self.sess = tf.Session()
            self.saver = tf.train.Saver()
            self.sess.run(self.init)
            self.saver.restore(self.sess, loc)
            self.action_out = self.graph.get_operation_by_name('base/action/Tanh').outputs[0]

        if method == '1':
            self.AE = AE()

        self.network = network
        self.method = method
        self.image_size = image_size
        self.dim_a = dim_a
        self.action_lower_limits = str_2_array(action_lower_limits)
        self.action_upper_limits = str_2_array(action_upper_limits)
        self.error_prob = float(error_prob)

        self.thr0 = 0.52  # 0.2
        self.thr1 = 0.26  # 0.07
        self.thr2 = 0.18  # 0.07

        self.teacher_parameters = self.get_teacher_parameters(exp)
        print('\nteacher parameters:', self.teacher_parameters)
        time.sleep(3)

    def action(self, observation):
        observation_gray = observation_to_gray(observation, self.image_size)
        observation = self.AE.conv_representation(observation_gray)

        action = self.sess.run(self.action_out, feed_dict={'base/input:0': observation})

        out_action = []
        for i in range(self.dim_a):
            action[0, i] = np.clip(action[0, i], self.action_lower_limits[i], self.action_upper_limits[i])
            out_action.append(action[0, i])

        return np.array(out_action)

    def get_feedback_signal(self, observation, agent_output, episode):
        action = self.action(observation)
        diff = action - agent_output

        feedback_prob = self.teacher_parameters[0]*np.exp(-self.teacher_parameters[1]*episode)

        error_prob = self.error_prob  # 0.07 -> 20%; 0.035 -> 10%

        h = np.array([0, 0])
        if np.random.uniform() < feedback_prob:
            h = np.sign(diff)

            # Give erroneous feedback
            if np.random.uniform() < error_prob:
                h[0] = h[0] * -1

            if np.random.uniform() < error_prob:
                h[1] = h[1] * -1

            if np.random.uniform() < error_prob:
                h[2] = h[2] * -1

        return h

    def new_episode(self, i_episode):
        return None

    def get_teacher_parameters(self, experiment):
        if experiment == '-1':
            return [0.6, 0.00001]
        parameters = {'1': [0.6, 0.00001], '2': [0.6, 0.00002], '3': [0.6, 0.00003], '4': [0.6, 0.00004],
                      '5': [0.6, 0.00005], '6': [0.6, 0.00006], '7': [0.6, 0.00007], '8': [0.6, 0.00008],
                      '9': [0.6, 0.00009], '10': [0.6, 0.0001], '11': [0.7, 0.00001], '12': [0.7, 0.00002],
                      '13': [0.7, 0.00003], '14': [0.7, 0.00004], '15': [0.7, 0.00005], '16': [0.7, 0.00006],
                      '17': [0.7, 0.00007], '18': [0.7, 0.00008], '19': [0.7, 0.00009], '20': [0.7, 0.0001],
                      '21': [0.8, 0.00001], '22': [0.8, 0.00002], '23': [0.8, 0.00003], '24': [0.8, 0.00004],
                      '25': [0.8, 0.00005], '26': [0.8, 0.00006], '27': [0.8, 0.00007], '28': [0.8, 0.00008],
                      '29': [0.8, 0.00009], '30': [0.8, 0.0001]}

        return [0.6, 0.000015]  # 0.000025

