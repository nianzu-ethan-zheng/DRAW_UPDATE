import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.layers as ly
import numpy as np
from sequential.utils import pickle_save, check_dir
from sequential.process import Process


NB_TEST_DATA = 10000


def mlp(_input, out_dims, name='mlp', reuse=False):
    """Multi-layer perception with single layer
    """
    with tf.variable_scope(name, reuse=reuse):
        _, n = _input.get_shape().as_list()
        W = tf.get_variable(
            name="weight",
            shape=[n, out_dims],
            initializer=ly.variance_scaling_initializer())
        b = tf.get_variable(
            name="bias",
            initializer=tf.constant(0.0, shape=[out_dims]))
        out = tf.matmul(_input, W) + b
        return out


class autoencoder(object):
    """Build a deep autoencoder
    """
    # %% input to the network
    def __init__(self, dimensions):
        self.dim = dimensions

    # %% Build the encoder
    def encoder(self, x, u, reuse=False):
        current = x
        z, mean, log_sigma, Lz = [], [], [], []
        for layer_i, n_output in enumerate(self.dim[1:]):
            if layer_i != len(self.dim[1:])-1:
                current = tf.nn.tanh(mlp(current, n_output, name="encoder_{}".format(layer_i), reuse=reuse))
            else:
                mean = mlp(current, n_output, name="encoder_{}u".format(layer_i), reuse=reuse)
                log_sigma = mlp(current, n_output, name="encoder_{}s".format(layer_i), reuse=reuse)
                z = mean + tf.exp(log_sigma) * u

                Lz = tf.reduce_mean(
                    tf.reduce_sum(- log_sigma + 0.5 * (tf.square(mean) + tf.exp(2*log_sigma)) - 0.5, axis=-1))
        return z, Lz

    # Latent representation (embedding, neural coding)

    def decoder(self, z, reuse=False):
        current = z
        # Build the decoder using the same weights
        for layer_i, n_output in enumerate(self.dim[:-1][::-1]):
            if layer_i != len(self.dim[:-1])-1:
                current = tf.nn.tanh(mlp(current, n_output, name="decoder_{}".format(layer_i), reuse=reuse))
            else:
                current = tf.nn.sigmoid(mlp(current, n_output, name="decoder_{}".format(layer_i), reuse=reuse))
        y = current
        return y

    def binary_crossentropy(self, t, o):
        eps = 1E-8
        return -(t * tf.log(o + eps) + (1.0 - t) * tf.log(1.0 - o + eps))


def train_autoencoder():

    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    dims = [784, 256, 120]
    batch_size = 100
    n_epochs = 100
    ae = autoencoder(dims)

    x = tf.placeholder(tf.float32, [batch_size, dims[0]], name='x')
    u = tf.random_normal([batch_size, dims[-1]], mean=0, stddev=1)
    z, Lz = ae.encoder(x, u)
    y = ae.decoder(z)
    Lx = tf.reduce_mean(tf.reduce_sum(ae.binary_crossentropy(x, y), 1))
    cost = Lx + Lz

    us = tf.random_normal([batch_size, dims[-1]], mean=0, stddev=1)
    ys = ae.decoder(us, reuse=True)

    learning_rate = 4E-3
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    process = Process()
    train_epoch_loss = []
    # Fit all training data
    for epoch_i in range(n_epochs):
        training_iteration_loss = []
        process.start_epoch()
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            tl, lx, lz, _ = sess.run([cost, Lx, Lz, optimizer], feed_dict={x: batch_xs})
            training_iteration_loss.append([tl, lx, lz])
        average_loss = np.mean(np.array(training_iteration_loss), axis=0)

        train_epoch_loss.append(average_loss)
        if epoch_i % 10 == 0:
            process.format_meter(epoch_i, n_epochs,
                                 {"cost": average_loss[0], "Lx": average_loss[1], "Lz": average_loss[2]})
    pickle_save(train_epoch_loss, ["loss"], "./result/loss.pkl")

    # Get embeddings.
    # If you have too much to get and that it does not fit in memory, you may
    # need to use a batch size or to force to use the CPU rather than the GPU.
    batch_xt, _ = mnist.train.next_batch(batch_size)
    x_recons = sess.run(y, feed_dict={x: batch_xt})

    # Get test samples
    x_sample = sess.run(ys)

    check_dir("./result")
    pickle_save([x_sample, x_recons], ["Generate", "Train"], "./result/ae.pkl")


if __name__ == '__main__':
    train_autoencoder()