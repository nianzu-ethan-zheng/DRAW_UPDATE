from sequential.attention import *
import tensorflow as tf
import os
from tensorflow.examples.tutorials import mnist
import numpy as np
import sys
print(sys.path)
print(os.getcwd())

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

center_y = 5
center_x = 14
delta = 2
sigma2 = 0
gamma = 0
N = 5

out = np.array([center_y, center_x,delta, sigma2, gamma]).reshape(1, -1)
print(out.shape)
l = tf.placeholder(tf.float32, [None, 5])
att = ZoomableAttetionWindow(channels, height, width, N)
center_y, center_x, delta, sigma2, gamma = att.nn2att(l, is_test=True)
W, region_r = att.read(x, center_y, center_x, delta, sigma2)
Img_, region_w = att.write(W, center_y, center_x, delta, sigma2)

sess = tf.InteractiveSession()
(W, region_r) = sess.run((W, region_r), feed_dict={x:x_batch, l:out})
print(W.shape, region_r)

I2, region_w = sess.run([Img_, region_w], feed_dict={x:x_batch, l:out})
print(region_w)

def imagify(flat_image, h, w):
    image = flat_image.reshape([h, w])
    image = image.transpose([0, 1])
    return image/ image.max()

import pylab
import matplotlib.patches as patches
fig, ax = pylab.subplots(1)
ax.imshow(imagify(x_batch, height, width), interpolation='nearest')
rect = patches.Rectangle(region_r[0, 0:2], region_r[0, 2], region_r[0, 3], color='r', fill=False)
ax.add_patch(rect)
#
pylab.figure()
pylab.gray()
pylab.imshow(imagify(W, N, N), interpolation='nearest')
#
fig2, ax = pylab.subplots(1)
ax.imshow(imagify(I2, height, width), interpolation='nearest')
rect = patches.Rectangle(region_w[0, 0:2], region_w[0, 2], region_w[0, 3], color='r', fill=False)
ax.add_patch(rect)
#
pylab.figure()
pylab.gray()
pylab.imshow(imagify(x_batch, height, width)-imagify(I2, height, width), interpolation='nearest')

pylab.show(block=True)