import tensorflow.contrib.layers as lays
import tensorflow as tf


def fully_connected_layers(encoder_output, dim_a, loss_function_type):
    in_shape = encoder_output.get_shape()
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(encoder_output, shape=[-1, in_shape[1], in_shape[2], in_shape[3]])

    # Flatten the data to a 1-D vector for the fully connected layer
    fc1 = tf.contrib.layers.flatten(x)

    # Fully connected layer (in tf contrib folder for now)
    fc1 = tf.layers.dense(fc1, 300, activation=tf.nn.tanh)
    fc1 = tf.layers.dense(fc1, 300, activation=tf.nn.tanh)

    # Output layer, class prediction
    y = tf.layers.dense(fc1, dim_a, activation=tf.nn.tanh, name='action')

    y_ = tf.placeholder(tf.float32, [None, dim_a], name='label')

    # define the loss function
    if loss_function_type == 'cross_entropy':
        loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    elif loss_function_type == 'mean_squared':
        loss = 0.5 * tf.reduce_mean(tf.square(y - y_))
    else:
        loss = None
        print('No existing loss function was selected, please try mean_squared or cross_entropy')
        exit()

    return y, loss
