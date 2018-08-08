from sequential.model_component import *

class ARAAModel:
    def __init__(self, batch_size, img_size, z_size, dec_size, n_iter, reader,reader_x,encoder_rnn,
                 sampler, decoder_rnn, writer, discriminator_rnn, dis_dim,**kwargs):
        self.batch_size = batch_size
        self.img_size = img_size
        self.z_size = z_size
        self.dec_size = dec_size
        self.n_iter = n_iter
        self.reader = reader
        self.reader_x = reader_x
        self.encoder_rnn = encoder_rnn
        self.sampler = sampler
        self.decoder_rnn = decoder_rnn
        self.writer = writer
        self.dis_rnn = discriminator_rnn
        self.dis_size = dis_dim

    def reconstruct(self, x):
        """Reconstruction
        """
        T = self.n_iter
        batch_size = self.batch_size

        # Container for collecting the output result
        cs, kls, c_ws, xs = [0]*T, [0]*T, [0]*T, [0]*T
        us = [0]*T
        # Initial states
        c_prev = tf.negative(tf.ones((batch_size, self.img_size))) * 4
        h_dec_prev = tf.zeros((batch_size, self.dec_size))
        enc_state = self.encoder_rnn.zero_state(batch_size, tf.float32)
        dec_state = self.decoder_rnn.zero_state(batch_size, tf.float32)

        reuse = False
        for t in range(self.n_iter):
            with tf.variable_scope("encoder"):
                us[t] = tf.random_normal((batch_size, self.z_size), mean=0, stddev=1)
                x_hat = x - tf.nn.sigmoid(c_prev)
                patch, region_read = self.reader.apply(x, x_hat, h_dec_prev, reuse=reuse)
                h_enc, enc_state = self.encoder_rnn(
                    state=enc_state,
                    inputs=tf.concat([patch, h_dec_prev], axis=1))

                z, kl = self.sampler.sample(h_enc, us[t], reuse=reuse)

            with tf.variable_scope("decoder"):
                h_dec, dec_state = self.decoder_rnn(
                    state=dec_state,
                    inputs=z,
                )
                c_update, region_write = self.writer.apply(h_dec, reuse=reuse)
                c = c_prev + c_update

            # Store the tensor patches
            cs[t], kls[t], c_ws[t], xs[t] =\
                c, kl, region_write, tf.nn.sigmoid(c)

            # Update the recurrent parameters
            reuse = True
            c_prev, h_dec_prev = c, h_dec

        # kls: shape n_inter*(batch_size,)
        # x_recons = tf.nn.sigmoid(cs[-1])
        x_recons = tf.nn.sigmoid(cs[-1])
        kls = tf.add_n(kls)

        dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")
        enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")
        return x_recons, kls, xs, c_ws, dec_vars, enc_vars

    def sample(self):
        """Sample from model
        """
        T = self.n_iter
        xs, c_ws, us = [0]*T, [0]*T, [0]*T

        # Initial parameters
        c_prev = tf.negative(tf.ones((self.batch_size, self.img_size))) * 4
        dec_state = self.decoder_rnn.zero_state(self.batch_size, tf.float32)

        for t in range(T):
            us[t] = tf.random_normal((self.batch_size, self.z_size), mean=0, stddev=1)
            with tf.variable_scope("decoder"):
                z = self.sampler.sample_from_prior(us[t])
                h_dec, dec_state = self.decoder_rnn(
                    state=dec_state,
                    inputs=z)
                c_update, region_write = self.writer.apply(h_dec, reuse=True)
                c = c_prev + c_update

            # Store the results
            xs[t], c_ws[t] = tf.sigmoid(c), region_write

            # Update the state
            c_prev = c

        return xs, c_ws

    def discriminator(self, x, ru=False):
        with tf.variable_scope("discriminator"):
            reuse = ru

            dis_state = self.dis_rnn.zero_state(self.batch_size, tf.float32)
            h_dis = tf.zeros((self.batch_size, self.dis_size))

            for t in range(self.n_iter):
                r, c_r = self.reader_x.apply(x, h_dis, reuse=reuse)
                h_dis, dis_state = self.dis_rnn(
                    state=dis_state,
                    inputs=r)
                reuse = True

            reuse = ru
            out = mlp(h_dis, 1, name="dis_mlp", reuse=reuse)

        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        return out, vars



