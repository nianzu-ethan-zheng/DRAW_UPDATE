from draw.draw_model import *
import os
from tensorflow.examples.tutorials import mnist

z_dim = 100
batch_size = 1
channels, height, width = 1, 28, 28
x_dim = 28*28
dec_size = 256
read_N, write_N = 2, 5

u = tf.random_normal((batch_size, z_dim), mean=0, stddev=1)
x = tf.placeholder(tf.float32, (batch_size, x_dim))
h_dec_prev = tf.zeros((batch_size, dec_size))

data_directory = os.path.join("./", "mnist")
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train
x_batch, _ = train_data.next_batch(batch_size)

#  Qsampler examination
# qsampler = Qsampler(z_dim)
# z, kl = qsampler.sample(x, u, reuse=False)
# print(z.get_shape(), kl.get_shape())
# print()
# z0 = qsampler.sample_from_prior(u)
# print(z0.get_shape())

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# z0, z, kl = sess.run([z0, z, kl], feed_dict={x : x_batch})
# print(z0, "/n", z, "/n", kl)

# Test reader and writer
# l = mlp(h_dec_prev, out_dims=5, name="read_parameter", reuse=False)
# print(l.shape)
# att = ZoomableAttetionWindow(channels, height, width, write_N)
# center_y, center_x, delta, sigma2, gamma = att.nn2att(l, is_test=False)
# W, region_r = att.read(x, center_y, center_x, delta, sigma2)
# Img_, region_w = att.write(W, center_y, center_x, delta, sigma2)
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# (W, region_r) = sess.run((W, region_r), feed_dict={x:x_batch})
# print(W.shape, region_r)
# center_y, center_x, delta, sigma2, gamma = sess.run([center_y, center_x, delta, sigma2, gamma])
#
# I2, region_w = sess.run([Img_, region_w], feed_dict={x:x_batch})
# print(region_w)
#
# def imagify(flat_image, h, w):
#     image = flat_image.reshape([h, w])
#     image = image.transpose([0, 1])
#     return image/ image.max()
#
# import pylab
# import matplotlib.patches as patches
# fig, ax = pylab.subplots(1)
# ax.imshow(imagify(x_batch, height, width), interpolation='nearest')
# rect = patches.Rectangle(region_r[0, 0:2], region_r[0, 2], region_r[0, 3], color='r', fill=False)
# ax.add_patch(rect)
# #
# pylab.figure()
# pylab.gray()
# pylab.imshow(imagify(W, write_N, write_N), interpolation='nearest')
# #
# fig2, ax = pylab.subplots(1)
# ax.imshow(imagify(I2, height, width), interpolation='nearest')
# rect = patches.Rectangle(region_w[0, 0:2], region_w[0, 2], region_w[0, 3], color='r', fill=False)
# ax.add_patch(rect)
# #
# pylab.figure()
# pylab.gray()
# pylab.imshow(imagify(x_batch, height, width)-imagify(I2, height, width), interpolation='nearest')
#
# pylab.show(block=True)


# Reader examination
# reader = Reader()
# xc = reader.apply(x, x, h_dec_prev, reuse=False)
# print(xc.get_shape())
# readerA = AttentionReader(channels, height, width, read_N)
# w, rg = readerA.apply(x, x, h_dec_prev, reuse=False)
# print(w.get_shape(), rg.get_shape())
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# z0, z, kl = sess.run([xc, w, rg], feed_dict={x : x_batch})
# print(z0, "/n", z, "/n", kl)


# # Writer examination
# writer = Writer(x_dim=x_dim)
# c = writer.apply(h_dec_prev)
# print(c.get_shape())
#
# Awriter = AttentionWriter(channels, height, width, write_N)
# ca, rg = Awriter.apply(h_dec_prev, reuse=False)
# print(c.get_shape(), rg.get_shape())
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# c, ca, kl = sess.run([c, ca, rg], feed_dict={x : x_batch})
# print(c, "/n", ca, "/n", kl)

# # DRAW MODEL
# n_iter, enc_dim, dec_dim= 5, 256, 256
# read_dim = 2*channels*read_N**2
#
# reader = AttentionReader(channels, height, width, read_N)
# writer = AttentionWriter(channels, height, width, write_N)
# encoder_rnn = tf.nn.rnn_cell.LSTMCell(enc_dim, state_is_tuple=True)
# decoder_rnn = tf.nn.rnn_cell.LSTMCell(dec_dim, state_is_tuple=True)
# sampler = Qsampler(z_dim)
#
# draw = DrawModel(
#         batch_size, x_dim, z_dim, dec_dim,
#         n_iter=n_iter,
#         reader=reader,
#         encoder_rnn=encoder_rnn,
#         sampler=sampler,
#         decoder_rnn=decoder_rnn,
#         writer=writer)
#
# c_prev = tf.zeros((batch_size, x_dim))
# enc_state_prev = encoder_rnn.zero_state(batch_size, tf.float32)
# dec_state_prev = decoder_rnn.zero_state(batch_size, tf.float32)
#
# # c, enc_state, kl, h_dec, dec_state, c_r, c_w = draw.iterate_body(u, c_prev, enc_state_prev,
# #                                                                  h_dec_prev, dec_state_prev, x, reuse=False)
# # print(c.get_shape(), enc_state, kl.get_shape(),
# #       h_dec.get_shape(), dec_state, c_r.get_shape(),
# #       c_w.get_shape())
#
# x_logits, kls = draw.reconstruct(x)
# print(x_logits.get_shape(), kls.get_shape())
# # Look at the parameters
# train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
# [print(var) for var in train_vars]

# One iteration of generate body
# c, dec_state, c_w = draw.generate_body(u, c_prev, dec_state_prev, reuse=True)
# print(c.get_shape(), dec_state, c_w.get_shape())

# xs, c_ws = draw.sample()
# print(xs, c_ws)
