"""
@Author: Nianzu-Ethan-Zheng
Address: Shenyang  NEU
Draw: model one iteration
        Parameters
        -----------------------------------------------------------------------------------------
        Parameter          Shape                                       Function
        -----------------------------------------------------------------------------------------
        u                  shape(batch_size, z_size),                  normal variables
        c_prev             shape(batch_size, channels*height*width)    canvas matrix
        enc_state_prev     shape(batch_size, enc_out_size)x2           state matrix of encoder
                           (h_enc_prev, cell_enc_prev)
        h_dec_prev         shape(batch_size, out_size)                 output of decoder
        dec_state_prev     shape(batch_size, dec_out_size)x2           state matrix of decoder
                           (h_dec_prev, cell_dec_prev)
        x                  shape(batch_size, channels*height*width)    Truth
        -----------------------------------------------------------------------------------------

        Return
        -----------------------------------------------------------------------------------------
        Parameter          Shape                                  Function
        -----------------------------------------------------------------------------------------
        c                  (batch_size, channels*height*width)    canvas variables
        enc_state          (batch_size, enc_out_size)x2           state matrix of encoder
        kl                 (batch_size, )                         Kullback-Leiber divergence
        h_dec              (batch_size, out_size)                 output of decoder
        dec_state          (batch_size, dec_out_size)x2           state matrix of decoder
        c_r                (batch_size, 4)                        Reading region
        c_w                (batch_size, 4)                        writing region
        ------------------------------------------------------------------------------------------
        """

from sequential.attention import ZoomableAttetionWindow
import tensorflow as tf
import tensorflow.contrib.layers as ly


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


class Qsampler():
    def __init__(self, z_dim):
        self.prior_mean = 0
        self.prior_log_sigma = 0.
        self.z_dim = z_dim

    def sample(self, x, u, reuse):
        """Return a samples and the corresponding KL term
        """
        mean = mlp(x, self.z_dim, name='e_mean_mlp', reuse=reuse)
        log_sigma = mlp(x, self.z_dim, name='e_log_sigma_mlp', reuse=reuse)

        # Sample from mean_zero, std.-one Gaussian
        # u = tf.random.normal(
        # (batch_size, z_size),
        # mean=0,
        # stddev=1)
        z = mean + tf.exp(log_sigma) * u

        # Calculate KL
        # URL: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        kl = tf.reduce_sum((self.prior_log_sigma - log_sigma +
             (tf.exp(2 * log_sigma) + (mean - self.prior_mean)**2) /
                            (2 * tf.exp(2 * self.prior_log_sigma)) - 0.5), axis=-1)

        return z, kl

    def sample_from_prior(self, u):
        z = self.prior_mean + tf.exp(self.prior_log_sigma) * u
        return z

# ----------------------------------------------------------------------------------


class Reader():
    def __init__(self, channels, height, width):
        self.channels = channels
        self.height = height
        self.width = width

    def apply(self, x, x_hat, h_dec, reuse=False):
        """Reader without attention
        """
        rg = tf.constant([0, 0, self.width, self.height])
        return tf.concat([x, x_hat], axis=1), rg


class AttentionReader():
    def __init__(self, channels, height, width, N):
        self.img_height = height
        self.img_width = width
        self.N = N
        self.output_dim = 2*channels*N*N

        self.zoomer = ZoomableAttetionWindow(channels, height, width, N)

    def apply(self, x, x_hat, h_dec, reuse=False):
        """Attention Reader
        """

        l = mlp(h_dec, out_dims=5, name="e_read_parameter", reuse=reuse)

        center_y, center_x, delta, sigma2, gamma = self.zoomer.nn2att(l, is_test=False)

        # Parameters
        # w, w_hat: shape(batch_size, channels*N*N)
        # rg: position(left lower at coordinates, and Nx,Ny)
        w, rg = self.zoomer.read(x, center_y, center_x, delta, sigma2)
        w_hat, _ = self.zoomer.read(x_hat, center_y, center_x, delta, sigma2)

        w, w_hat = w * gamma, w_hat * gamma
        return tf.concat([w, w_hat], axis=1), rg

# ---------------------------------------------------------------------------------


class Writer:
    def __init__(self, x_dim, channels, height, width):
        self.out_dim = x_dim
        self.channels = channels
        self.height = height
        self.width = width

    def apply(self, h, reuse):
        rg = tf.constant([0, 0, self.width, self.height])
        out = mlp(h, self.out_dim, name="d_writer_transformer", reuse=reuse)
        return out, rg


class AttentionWriter():
    def __init__(self, channels, height, width, N):
        self.img_height = height
        self.img_width = width
        self.N = N
        self.out_dim = channels * N * N
        self.zoomer = ZoomableAttetionWindow(channels, height, width, N)

    def apply(self, h, reuse):
        l = mlp(h, out_dims=5, name="d_write_parameter", reuse=reuse)
        w = mlp(h, out_dims=self.out_dim, name="d_write_w", reuse=reuse)

        center_y, center_x, delta, sigma2, gamma = self.zoomer.nn2att(l, is_test=False)

        # Parameters
        # w, w_hat: shape(batch_size, channels*N*N)
        # rg: position(left lower at coordinates, and Nx,Ny)
        c_update, rg = self.zoomer.write(w, center_y, center_x, delta, sigma2)
        c_update *= 1 / gamma

        return c_update, rg

# ---------------------------------------------------
class Reader_x():
    def __init__(self, channels, height, width):
        self.channels = channels
        self.height = height
        self.width = width

    def apply(self, x, h_dec, reuse=False):
        """Reader without attention
        """
        rg = tf.constant([0, 0, self.width, self.height])
        return x, rg

class AttentionReader_x():
    def __init__(self, channels, height, width, N):
        self.img_height = height
        self.img_width = width
        self.N = N
        self.output_dim = 2*channels*N*N

        self.zoomer = ZoomableAttetionWindow(channels, height, width, N)

    def apply(self, x, h_dec, reuse=False):
        """Attention Reader
        """

        l = mlp(h_dec, out_dims=5, name="e_read_parameter", reuse=reuse)

        center_y, center_x, delta, sigma2, gamma = self.zoomer.nn2att(l, is_test=False)

        # Parameters
        # w, w_hat: shape(batch_size, channels*N*N)
        # rg: position(left lower at coordinates, and Nx,Ny)
        w, rg = self.zoomer.read(x, center_y, center_x, delta, sigma2)

        w *= gamma
        return w, rg