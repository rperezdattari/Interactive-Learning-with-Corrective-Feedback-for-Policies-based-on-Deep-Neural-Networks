import numpy as np
import matplotlib.pyplot as plt
import configparser


def load_config_data(config_dir):
    config = configparser.ConfigParser()
    config.read(config_dir)
    return config


def str_2_array(str_state_shape, type_n='int'):
    sep_str_state_shape = str_state_shape.split(',')
    state_n_dim = len(sep_str_state_shape)
    state_shape = []
    for i in range(state_n_dim):
        if type_n == 'int':
            state_shape.append(int(sep_str_state_shape[i]))
        elif type_n == 'float':
            state_shape.append(float(sep_str_state_shape[i]))
        else:
            print('Selected type for str_2_array not implemented.')
            exit()

    return state_shape


def observation_to_gray(observation, image_size):
    observation = np.array(observation).reshape(1, image_size, image_size, 3)
    observation_gray = np.mean(observation, axis=3)
    observation_gray = observation_gray.reshape(
        (-1, image_size, image_size, 1))
    observation_gray_norm = observation_gray / 255.0

    return observation_gray_norm


class FastImagePlot:
    def __init__(self, fig_num, observation, image_size, title_name, vmin=0, vmax=1):
        self.window = plt.figure(fig_num)
        self.image_size = image_size
        self.im = plt.imshow(np.reshape(observation, [self.image_size, self.image_size]),
                             cmap='gray', vmin=vmin, vmax=vmax)
        plt.show(block=False)
        self.window.canvas.set_window_title(title_name)
        self.window.canvas.draw()

    def refresh(self, observation):
        self.im.set_data(np.reshape(observation, [self.image_size, self.image_size]))
        self.window.draw_artist(self.im)
        self.window.canvas.blit()
        self.window.canvas.flush_events()
