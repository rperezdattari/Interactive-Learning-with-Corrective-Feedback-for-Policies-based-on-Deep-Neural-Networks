import tensorflow as tf
import numpy as np
from tools.functions import observation_to_gray, str_2_array
from autoencoder import AE
import time
import cv2


class Teacher:
    def __init__(self, image_size=64, dim_a=3,
                 action_lower_limits='0,0,0', action_upper_limits='1,1,1',
                 loc='graphs/teacher/CarRacing-v0/network', exp='1', error_prob=0,
                 resize_observation=True, teacher_parameters='0.6,0.00001'):
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

        self.AE = AE()

        self.image_size = image_size
        self.dim_a = dim_a
        self.action_lower_limits = str_2_array(action_lower_limits)
        self.action_upper_limits = str_2_array(action_upper_limits)
        self.error_prob = float(error_prob)
        self.resize_observation = resize_observation

        self.teacher_parameters = str_2_array(teacher_parameters, 'float')
        print('\nTeacher parameters:', self.teacher_parameters)
        time.sleep(3)

    def action(self, observation):
        if self.resize_observation:
            observation = cv2.resize(observation, (self.image_size, self.image_size))
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

