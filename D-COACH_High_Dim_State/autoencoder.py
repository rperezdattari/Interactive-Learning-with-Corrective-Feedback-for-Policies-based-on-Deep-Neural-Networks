from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class TrainAE:
    """Class in charge of training and testing the autoencoder."""
    def __init__(self):
        self.batch_size = 100  # Number of samples in each batch
        self.epoch_num = 20     # Number of epochs to train the network
        self.lr = 0.001        # Learning rate
        self.train_AE = True
        self.use_pre_trained_weights = False
        self.save_weights = True
        self.weight_loc = 'graphs/autoencoder/conv_layers_64x64'
        self.database = np.load('racing_car_classic_database_64x64.npy')

    def next_batch(self, database, batch_size):
        batch = database[np.random.choice(len(database), size=batch_size, replace=False)]
        batch = np.mean(batch, axis=3)
        for k in range(len(batch)):
            batch[k, :, :] = batch[k, :, :] / 255.0
        return batch

    def autoencoder(self):
        self.ae_inputs = tf.placeholder(tf.float32, (None, 64, 64, 1), name='image')
        # encoder
        net = lays.conv2d(self.ae_inputs, 16, [3, 3], stride=2, padding='SAME')
        net = lays.conv2d(net, 8, [3, 3], stride=2, padding='SAME')
        net = lays.conv2d(net, 4, [3, 3], stride=2, padding='SAME')
        net = tf.identity(net, name='conv_part')

        # decoder
        net = lays.conv2d_transpose(net, 8, [3, 3], stride=2, padding='SAME')
        net = lays.conv2d_transpose(net, 16, [3, 3], stride=2, padding='SAME')
        net = lays.conv2d_transpose(net, 1, [3, 3], stride=2, padding='SAME', activation_fn=tf.nn.sigmoid)
        self.ae_output = tf.identity(net, name='ae_output')

        # calculate the loss and optimize the network
        self.loss = tf.reduce_mean(tf.square(self.ae_output - self.ae_inputs))
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def run(self, train=True, show_performance=True):
        # calculate the number of batches per epoch
        batch_per_ep = 12000 // self.batch_size
        graph = tf.Graph()
        sess = tf.Session(graph=graph)

        with graph.as_default():
            self.autoencoder()  # create the Autoencoder network
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess.run(init)

            if self.use_pre_trained_weights:
                saver.restore(sess, self.weight_loc)

            if train:
                for ep in range(self.epoch_num):  # epochs loop
                    for batch_n in range(batch_per_ep):  # batches loop
                        batch_img = self.next_batch(self.database, self.batch_size)  # read a batch
                        batch_img = batch_img.reshape((-1, 64, 64, 1))               # reshape each sample to an (96, 96) image
                        _, c, outputs = sess.run([self.train_op, self.loss, self.ae_output], feed_dict={self.ae_inputs: batch_img})
                        print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))
                        print('Batch progress:', batch_n/batch_per_ep * 100, '%')

                        if self.save_weights:
                            saver.save(sess, self.weight_loc)

            if show_performance:
                # test the trained network
                batch_img = self.next_batch(self.database, self.batch_size)  # read a batch
                batch_img = batch_img.reshape((-1, 64, 64, 1))  # reshape each sample to an (28, 28) image
                recon_img = sess.run([self.ae_output], feed_dict={self.ae_inputs: batch_img})[0]

                # plot the reconstructed images and their ground truths (inputs)
                plt.figure(1)
                plt.title('Reconstructed Images')
                for i in range(5):
                    plt.subplot(1, 5, i+1)
                    plt.imshow(recon_img[i, ..., 0], cmap='gray')
                plt.figure(2)
                plt.title('Input Images')
                for i in range(5):
                    plt.subplot(1, 5, i+1)
                    plt.imshow(batch_img[i, ..., 0], cmap='gray')
                plt.show()


class AE:
    """Class in charge of importing the trained weights of the
    autoencoder with its corresponding graph and evaluate it."""
    def __init__(self, ae_loc='graphs/autoencoder/CarRacing-v0/conv_layers_64x64'):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(ae_loc + '.meta', clear_devices=True)
            self.init = tf.global_variables_initializer()  # initialize the graph
            self.sess = tf.Session()
            self.saver = tf.train.Saver()
            self.sess.run(self.init)
            self.saver.restore(self.sess, ae_loc)
            self.code = self.graph.get_operation_by_name('conv_part').outputs[0]
            self.ae_output = self.graph.get_operation_by_name('ae_output').outputs[0]

    def conv_representation(self, observation):
        return self.sess.run(self.code, feed_dict={'image:0': observation})

    def output(self, observation):
        return self.sess.run(self.ae_output, feed_dict={'image:0': observation})


if __name__ == "__main__":
    AE = TrainAE()
    AE.run(train=True, show_performance=True)
