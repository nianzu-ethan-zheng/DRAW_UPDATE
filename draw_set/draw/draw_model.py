from sequential.model_component import *


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
        cs, kls, c_ws, xs, zs, us = [0]*T, [0]*T, [0]*T, [0]*T, [0]*T, [0]*T

        # Initial states
        c_prev = tf.negative(tf.ones((batch_size, self.img_size))) * 8
        h_dec_prev = tf.zeros((batch_size, self.dec_size))
        enc_state_prev = self.encoder_rnn.zero_state(batch_size, tf.float32)
        dec_state_prev = self.decoder_rnn.zero_state(batch_size, tf.float32)
        reuse = False

        for t in range(self.n_iter):
            us[t] = tf.random_normal((batch_size, self.z_size), mean=0, stddev=1)
            c, enc_state, kl, h_dec, dec_state, _, c_w , z = self.iterate_body(
            us[t], c_prev, enc_state_prev, h_dec_prev, dec_state_prev, x, reuse=reuse)

            # Store the tensor patches
            cs[t], kls[t], c_ws[t], xs[t], zs[t] = c, kl, c_w, tf.nn.sigmoid(c), z

            # Update the recurrent parameters
            reuse = True
            c_prev, enc_state_prev, h_dec_prev, dec_state_prev = \
                c, enc_state, h_dec, dec_state

        # kls: shape n_inter*(batch_size,)
        # x_recons = tf.nn.sigmoid(cs[-1])
        x_recons = cs[-1]
        kls = tf.add_n(kls)

        return x_recons, kls, xs, c_ws, zs

    def sample(self):
        """Sample from model
        """
        T = self.n_iter
        xs, c_ws, zs, us = [0]*T, [0]*T, [0]*T, [0]*T
        # Initial parameters
        c_prev = tf.negative(tf.ones((self.batch_size, self.img_size))) * 8
        dec_state_prev = self.decoder_rnn.zero_state(self.batch_size, tf.float32)

        for t in range(T):
            us[t] = tf.random_normal((self.batch_size, self.z_size), mean=0, stddev=1)
            c, dec_state, c_w, z = \
                self.generate_body(us[t], c_prev, dec_state_prev, reuse=True)

            # Store the results
            xs[t], c_ws[t], zs[t] = tf.sigmoid(c), c_w, z

            # Update the state
            c_prev, dec_state_prev = c, dec_state

        return xs, c_ws, zs

    def iterate_body(self, u, c_prev, enc_state_prev, h_dec_prev, dec_state_prev, x, reuse=False):
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

        return c, enc_state, kl, h_dec, dec_state, c_r, c_w, z

    def generate_body(self, u, c_prev, dec_state_prev, reuse=True):
        """Stochastic data generation
        """
        z = self.sampler.sample_from_prior(u)

        h_dec, dec_state = self.decoder_rnn(state=dec_state_prev, inputs=z)

        # Update the canvas matrix
        c_update, c_w = self.writer.apply(h_dec, reuse=reuse)
        c = c_prev + c_update
        return c, dec_state, c_w, z