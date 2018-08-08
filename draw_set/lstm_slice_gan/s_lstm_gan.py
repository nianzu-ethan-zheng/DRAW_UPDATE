from sequential.model_component import *

## STATE VARIABLES ##

class LstmModel:
    def __init__(self, batch_size, img_size, z_size, dec_size, n_iter,
                 sampler, decoder_rnn, discriminator_rnn, dis_size, patch_size,**kwargs):
        self.batch_size = batch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.z_size = z_size
        self.dec_size = dec_size
        self.n_iter = n_iter
        self.decoder_rnn = decoder_rnn
        self.dis_rnn = discriminator_rnn
        self.dis_size = dis_size

    def discriminator(self, ps, reuse=False):
        with tf.variable_scope("dis_rnn"):
            dis_state = self.dis_rnn.zero_state(self.batch_size, tf.float32)

            h_dis, _ = tf.nn.dynamic_rnn(
                cell=self.dis_rnn,
                inputs=ps,
                initial_state=dis_state,
                dtype=tf.float32,
                time_major=True)

            out = mlp(h_dis[-1], 1, name="dis_mlp", reuse=reuse)

        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="dis_rnn")
        return out, vars

    def generator(self):
        us = [0] * self.n_iter
        cs = [0] * self.n_iter

        with tf.variable_scope("gen_rnn"):
            gen_state = self.decoder_rnn.zero_state(self.batch_size, tf.float32)
            reuse = False
            for t in range(self.n_iter):
                c_prev = tf.negative(tf.ones((self.batch_size, self.patch_size))) * 4 if t==0 else cs[t-1]
                us[t] = tf.random_normal((self.batch_size, self.z_size), mean=0, stddev=1)
                h_gen, gen_state = self.decoder_rnn(state=gen_state, inputs=tf.concat([us[t], c_prev], axis=1))

                cs[t] = mlp(h_gen, self.patch_size,reuse=reuse)

                reuse = True

        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gen_rnn")
        return tf.nn.tanh(cs), vars











