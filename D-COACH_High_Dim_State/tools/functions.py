import numpy as np


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
