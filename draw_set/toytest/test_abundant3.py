import tensorflow as tf
import numpy as np

bs = 20
epochs = 1000

# dataset creation
xs = np.arange(0, 1, 0.01)
xs = np.expand_dims(xs, axis=1)
z = 0.2 * xs


class dataIterator:
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def next_batch(self, num):
        idx = np.arange(len(self.label))
        np.random.shuffle(idx)
        idx = idx[:num]
        return self.data[idx, :], self.label[idx]

# create model
x = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
w = tf.get_variable("w", shape=[1, 1], initializer=tf.constant_initializer(0.1))
b = tf.get_variable("b", shape=[], initializer=tf.constant_initializer(0))

out = tf.matmul(x, w) + b

loss = tf.reduce_mean(tf.square(y-out))

# build optimizer
optim = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# train
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

data = dataIterator(x=xs, y=z)

for epoch in range(epochs):
    x_batch, y_batch = data.next_batch(bs)
    # l, _ = sess.run([loss, optim], feed_dict={x: xs, y: z})
    l, _ = sess.run([loss, optim], feed_dict={x: x_batch, y: y_batch})
    print("the loss is {}".format(l))

print("the weight finally is: \n", sess.run(w))
zp = sess.run(out, feed_dict={x: xs})

import matplotlib.pyplot as plt

plt.plot(xs, z, "*r-")
plt.plot(xs, zp, ".-b")
plt.legend(["truth", "prediction"])
plt.show()






