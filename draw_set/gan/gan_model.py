import tensorflow as tf
import tensorflow.contrib.layers as ly


def mlp(_input, out_dims, name='mlp', std=0.05,reuse=False):
    """Multi-layer perception with single layer
    """
    with tf.variable_scope(name, reuse=reuse):
        _, n = _input.get_shape().as_list()
        W = tf.get_variable(
            name="weight",
            shape=[n, out_dims],
            initializer=tf.random_normal_initializer(stddev=std))
        b = tf.get_variable(
            name="bias",
            shape = [out_dims],
            initializer=tf.constant_initializer(0.0))
        out = tf.matmul(_input, W) + b
        return out


def maxout(x, num_pieces, num_units, name="maxout", std=0.05, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        d = x.get_shape().as_list()[-1]
        W = tf.get_variable(name="weight",
                            shape=[d, num_units, num_pieces],
                            initializer=tf.random_normal_initializer(stddev=std))
        b = tf.get_variable(name="bias",
                            initializer=tf.constant(0.0, shape=[num_units, num_pieces]))
        z = tf.tensordot(x, W, axes=1) + b
        z = tf.reduce_max(z, axis=2)
    return z


class Generator(object):
    def __init__(self, dim):
        self.name="generator"
        self.dim = dim
        self.out_dim = 784

    def __call__(self, z):
        with tf.variable_scope(self.name) as scope:
            g = tf.nn.relu(mlp(z, 256, name="g1"))
            # g = tf.nn.leaky_relu(mlp(g, 1200, name="g2"))
            # g = tf.nn.leaky_relu(mlp(g, 1024, name="g3"))
            g = tf.nn.tanh(mlp(g, self.out_dim, name="g4"))
        return g

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Discriminator(object):
    def __init__(self, dim):
        self.name = "discriminator"
        self.h_dim = 240

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name):
            d = tf.nn.relu(mlp(x, 256, name="d1", reuse=reuse))
            d = tf.nn.dropout(d, keep_prob=0.3)
            # d = tf.nn.relu(mlp(d, 512, name="d2", reuse=reuse))
            # d = tf.nn.dropout(d, keep_prob=0.3)
            # d = tf.nn.relu(mlp(d, 256, name="d3", reuse=reuse))
            # d = tf.nn.dropout(d, keep_prob=0.3)

            # d = maxout(x, num_pieces=5, num_units=self.h_dim, name="maxout1", reuse=reuse)
            # d = tf.nn.dropout(d, keep_prob=0.3)
            # d = maxout(d, num_pieces=5, num_units=self.h_dim, name="maxout2", reuse=reuse)
            # d = tf.nn.dropout(d, keep_prob=0.3)
            logit = mlp(d, 1, name="d4", reuse=reuse)
        return logit

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


# ------------------------------------------------------------------------------------------


