import tensorflow.contrib.layers as lays
import tensorflow as tf


def autoencoder(lr):
    ae_inputs = tf.placeholder(tf.float32, (None, 64, 64, 1), name='image')
    # encoder
    net = lays.conv2d(ae_inputs, 16, [3, 3], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [3, 3], stride=2, padding='SAME')
    net = lays.conv2d(net, 4, [3, 3], stride=2, padding='SAME')
    net = tf.identity(net, name='conv_part')

    # decoder
    net = lays.conv2d_transpose(net, 8, [3, 3], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 16, [3, 3], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 1, [3, 3], stride=2, padding='SAME', activation_fn=tf.nn.sigmoid)
    ae_output = tf.identity(net, name='ae_output')

    # calculate the loss and optimize the network
    loss = tf.reduce_mean(tf.square(ae_output - ae_inputs))
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    return loss, train_op, ae_inputs, ae_output


def fully_connected_layers(encoder_output, dim_a, fc_layers_neurons, loss_function_type):
    in_shape = encoder_output.get_shape()
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(encoder_output, shape=[-1, in_shape[1], in_shape[2], in_shape[3]])

    # Flatten the data to a 1-D vector for the fully connected layer
    fc1 = tf.contrib.layers.flatten(x)

    # Fully connected layer (in tf contrib folder for now)
    fc1 = tf.layers.dense(fc1, fc_layers_neurons, activation=tf.nn.tanh)
    fc1 = tf.layers.dense(fc1, fc_layers_neurons, activation=tf.nn.tanh)

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
