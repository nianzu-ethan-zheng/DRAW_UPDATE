from attention import ZoomableAttetionWindow
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
        Parameters
        --------------
        x: shape(Batch_size, encoder_size)

        Return
        --------------
        z: tensor.matrix
           Samples drawn from Q(z|x)
        KL: KL(Q(z|x)||P_z)
        """
        mean = mlp(x, self.z_dim, name='mean_mlp', reuse=reuse)
        log_sigma = mlp(x, self.z_dim, name='log_sigma_mlp', reuse=reuse)

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
        Parameters
        -----------
        x, x_hat : shape(batch_size, x_dim)

        Return
        ---------------
        out: shape(batch_size, 2*x_dim)
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
        ___________________________________________________________
        Variable        Shape                      Function
        -----------------------------------------------------------
        x, x_hat       (batch_size, x_dim)         truth, error image
        h_dec          (batch_size, dec_size)      output of decoder
        w, w_hat       (batch_size, channels*N*N)
        rg             (batch_size, 4)
        gamma          (batch_size, 1)
        ___________________________________________________________
        """

        l = mlp(h_dec, out_dims=5, name="read_parameter", reuse=reuse)

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
        out = mlp(h, self.out_dim, name="writer_transformer", reuse=reuse)
        return out, rg


class AttentionWriter():
    def __init__(self, channels, height, width, N):
        self.img_height = height
        self.img_width = width
        self.N = N
        self.out_dim = channels * N * N
        self.zoomer = ZoomableAttetionWindow(channels, height, width, N)

    def apply(self, h, reuse):
        l = mlp(h, out_dims=5, name="write_parameter", reuse=reuse)
        w = mlp(h, out_dims=self.out_dim, name="write_w", reuse=reuse)

        center_y, center_x, delta, sigma2, gamma = self.zoomer.nn2att(l, is_test=False)

        # Parameters
        # w, w_hat: shape(batch_size, channels*N*N)
        # rg: position(left lower at coordinates, and Nx,Ny)
        c_update, rg = self.zoomer.write(w, center_y, center_x, delta, sigma2)
        c_update *= 1 / gamma

        return c_update, rg

# ---------------------------------------------------------------------------------


class DrawModel:
    def __init__(self, batch_size, img_size,z_size, dec_size, n_iter, reader,encoder_rnn,
                 sampler, decoder_rnn, writer, **kwargs):
        self.batch_size = batch_size
        self.img_size = img_size
        self.z_size = z_size
        self.dec_size = dec_size
        self.n_iter = n_iter
        self.reader = reader
        self.encoder_rnn = encoder_rnn
        self.sampler = sampler
        self.decoder_rnn = decoder_rnn
        self.writer = writer

    def reconstruct(self, x):
        """Reconstruction
        """
        T = self.n_iter
        batch_size = self.batch_size

        # Container for collecting the output result
        cs, kls, c_ws, xs = [0]*T, [0]*T, [0]*T, [0]*T

        # Initial states
        c_prev = tf.negative(tf.ones((batch_size, self.img_size)))
        h_dec_prev = tf.zeros((batch_size, self.dec_size))
        enc_state_prev = self.encoder_rnn.zero_state(batch_size, tf.float32)
        u = tf.random_normal((batch_size, self.z_size), mean=0, stddev=1)
        dec_state_prev = self.decoder_rnn.zero_state(batch_size, tf.float32)
        reuse = False

        for t in range(self.n_iter):
            c, enc_state, kl, h_dec, dec_state, _, c_w = self.iterate_body(
            u, c_prev, enc_state_prev, h_dec_prev, dec_state_prev, x, reuse=reuse)

            # Store the tensor patches
            cs[t], kls[t], c_ws[t], xs[t] = c, kl, c_w, tf.nn.sigmoid(c)

            # Update the recurrent parameters
            reuse = True
            c_prev, enc_state_prev, h_dec_prev, dec_state_prev = \
                c, enc_state, h_dec, dec_state

        # kls: shape n_inter*(batch_size,)
        # x_recons = tf.nn.sigmoid(cs[-1])
        x_recons = cs[-1]
        kls = tf.add_n(kls)

        return x_recons, kls, xs, c_ws

    def sample(self):
        """Sample from model
        """
        T = self.n_iter
        xs, c_ws = [0]*T, [0]*T
        # Initial parameters
        u = tf.random_normal((self.batch_size, self.z_size), mean=0, stddev=1)
        c_prev = tf.zeros((self.batch_size, self.img_size))
        dec_state_prev = self.decoder_rnn.zero_state(self.batch_size, tf.float32)

        for t in range(T):
            c, dec_state, c_w = \
                self.generate_body(u, c_prev, dec_state_prev, reuse=True)

            # Store the results
            xs[t], c_ws[t] = tf.sigmoid(c), c_w

            # Update the state
            c_prev, dec_state_prev = c, dec_state

        return xs, c_ws

    def iterate_body(self, u, c_prev, enc_state_prev, h_dec_prev, dec_state_prev, x, reuse=False):
        """Draw: model one iteration
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
        # Get the error image
        x_hat = x - tf.nn.sigmoid(c_prev)

        # Determined by decoder_rnn precious performance,
        # the reader window scan the correlated regions.
        r, c_r = self.reader.apply(x, x_hat, h_dec_prev, reuse=reuse)

        # Attention window given by reader is produced by the encoder
        h_enc, enc_state = self.encoder_rnn(
            state=enc_state_prev,
            inputs=tf.concat([r, h_dec_prev], axis=1))

        # Sample from a particular distribution
        z, kl = self.sampler.sample(h_enc, u, reuse)

        h_dec, dec_state = self.decoder_rnn(
            state=dec_state_prev, inputs=z)

        # Update the canvas matrix
        c_update, c_w = self.writer.apply(h_dec, reuse=reuse)
        c = c_prev + c_update

        return c, enc_state, kl, h_dec, dec_state, c_r, c_w

    def generate_body(self, u, c_prev, dec_state_prev, reuse=True):
        """Stochastic data generation
        """
        z = self.sampler.sample_from_prior(u)

        h_dec, dec_state = self.decoder_rnn(state=dec_state_prev, inputs=z)

        # Update the canvas matrix
        c_update, c_w = self.writer.apply(h_dec, reuse=reuse)
        c = c_prev + c_update
        return c, dec_state, c_w