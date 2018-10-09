import numpy as np
from tools.functions import str_2_array


class MemoryBuffer:
    def __init__(self, buffer_length=1000, buffer_sampling_size=100, automatic_buffer_train=True, dim_a=3,
                 state_shape=1, network_state_shape=None):
        self.buffer_size = buffer_length
        self.buffer_sampling_size = buffer_sampling_size
        self.automatic_buffer_train = automatic_buffer_train
        self.state = []
        self.action = []
        self.network_state = []
        self.dim_a = dim_a
        self.state_shape = str_2_array(state_shape)
        self.network_state_shape = network_state_shape

    def append_state_ylabel(self, observation, y):
        self.state.append(observation)
        self.action.append(y)

    def get_batch(self):
        # Generate uniform random sample from data
        if len(self.action) < self.buffer_sampling_size:
            n_samples = int(len(self.action)/4) + 1
        else:
            n_samples = self.buffer_sampling_size
        if len(self.action) > self.buffer_size:
            self.state = self.state[-self.buffer_size:]
            self.action = self.action[-self.buffer_size:]
        sampled_buffer_l, sampled_action = self.random_sampling(self.state, self.action, n_samples)

        return sampled_buffer_l, sampled_action

    def is_long_enough(self):
        if len(self.action) > 50 and self.automatic_buffer_train:
            return True
        else:
            return False

    def random_sampling(self, state, action, n_samples):
        selected = np.random.choice(len(state), n_samples, replace=False)
        sampled_state = []
        sampled_action = []
        for i in selected:
            sampled_state.append(state[i].reshape(self.state_shape))
            sampled_action.append(action[i].reshape([self.dim_a]))

        return sampled_state, sampled_action

    def auto_training(self, last_episode_received_feedback):
        if last_episode_received_feedback > -1:
            self.automatic_buffer_train = True
        else:
            self.automatic_buffer_train = False
            self.state = []
            self.action = []
        #print('automatic buffer update state:', self.automatic_buffer_train)

    def get_feed_dict(self, **kwargs):
        feed_dict = {}
        for key, value in kwargs.items():
            feed_dict[key] = value

        return feed_dict