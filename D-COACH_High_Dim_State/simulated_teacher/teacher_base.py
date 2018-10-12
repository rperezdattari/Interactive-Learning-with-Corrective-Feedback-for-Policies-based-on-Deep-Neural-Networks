import tensorflow as tf
import numpy as np
from tools.functions import str_2_array
import time


class TeacherBase:
    def __init__(self, dim_a=3, action_lower_limits='0,0,0', action_upper_limits='1,1,1',
                 loc='graphs/teacher/CarRacing-v0/network', error_prob=0, teacher_parameters='0.6,0.00001'):

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

        self.low_dim_observation = None
        self.dim_a = dim_a
        self.action_lower_limits = str_2_array(action_lower_limits)
        self.action_upper_limits = str_2_array(action_upper_limits)
        self.error_prob = float(error_prob)
        self.teacher_parameters = str_2_array(teacher_parameters, 'float')

        print('\nTeacher parameters:', self.teacher_parameters)
        time.sleep(3)

    def _preprocess_observation(self, observation):
        pass

    def action(self, observation):
        self._preprocess_observation(observation)
        action = self.sess.run(self.action_out, feed_dict={'base/input:0': self.low_dim_observation})

        out_action = []
        for i in range(self.dim_a):
            action[0, i] = np.clip(action[0, i], self.action_lower_limits[i], self.action_upper_limits[i])
            out_action.append(action[0, i])

        return np.array(out_action)

    def get_feedback_signal(self, observation, agent_output, episode):
        action = self.action(observation)
        diff = action - agent_output

        feedback_prob = self.teacher_parameters[0]*np.exp(-self.teacher_parameters[1]*episode)

        h = np.array([0, 0])
        if np.random.uniform() < feedback_prob:
            h = np.sign(diff)

            # Give erroneous feedback
            for i in range(self.dim_a):
                if np.random.uniform() < self.error_prob:
                    h[i] = h[i] * -1

        return h
