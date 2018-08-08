from sequential.model_component import *

## STATE VARIABLES ##

class LstmModel:
    def __init__(self, batch_size, img_size, z_size, dec_size, n_iter, reader,encoder_rnn,
                 sampler, decoder_rnn, writer, discriminator_rnn, dis_size, **kwargs):
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
        self.dis_rnn = discriminator_rnn
        self.dis_size = dis_size

    def discriminator(self, x, ru=False):
        with tf.variable_scope("dis_rnn"):

            reuse = ru

            dis_state = self.dis_rnn.zero_state(self.batch_size, tf.float32)
            h_dis = tf.zeros((self.batch_size, self.dis_size))

            for t in range(self.n_iter):
                r, c_r = self.reader.apply(x, h_dis, reuse=reuse)
                h_dis, dis_state = self.dis_rnn(
                    state=dis_state,
                    inputs=r)
                reuse = True

            reuse = ru
            out = mlp(h_dis, 1, name="dis_mlp", reuse=reuse)

        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="dis_rnn")
        return out, vars

    def generator(self, c_int=None):
        us = [0] * self.n_iter
        cs = [0] * self.n_iter
        r_ws = [0] * self.n_iter

        with tf.variable_scope("gen_rnn"):
            gen_state = self.decoder_rnn.zero_state(self.batch_size, tf.float32)
            if c_int == None:
                c_int = tf.negative(tf.ones((self.batch_size, self.img_size))) * 0.5
                # c_int = tf.zeros([self.batch_size, self.img_size])
                # c_int = tf.ones((self.batch_size, self.img_size))

            reuse = False
            for t in range(self.n_iter):
                c_prev =  c_int if t==0 else cs[t-1]
                us[t] = tf.random_normal((self.batch_size, self.z_size), mean=0, stddev=1)
                h_gen, gen_state = self.decoder_rnn(state=gen_state, inputs=us[t])
                c_update, r_w = self.writer.apply(h_gen, reuse=reuse)

                cs[t] = c_prev + c_update
                reuse = True
                r_ws[t] = r_w

        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gen_rnn")
        return tf.nn.tanh(cs[-1]), vars, tf.nn.tanh(cs), r_ws











