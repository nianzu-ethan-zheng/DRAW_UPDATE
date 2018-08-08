import tensorflow as tf
import numpy as np

bs = 20
epochs = 1000

# dataset creation
xs = np.arange(0, 1, 0.01)
z = 0.2 * xs ** 2
t = np.random.random(xs.size)
i = np.stack([xs, t], axis=0).T

z = np.expand_dims(z, axis=1)


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
x = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
w = tf.get_variable("w", shape=[2, 1], initializer=tf.variance_scaling_initializer)
b = tf.get_variable("b", shape=[], initializer=tf.constant_initializer(0))

# w_ = tf.Print(w, [w], message="the weight is:\t")
# b_ = tf.Print(b, [b], message="the bias is:\t")

out = tf.matmul(x, w) + b

print(out.get_shape)

loss = tf.reduce_mean(tf.square(y-out))

# build optimizer
optim = tf.train.AdamOptimizer(learning_rate=0.5).minimize(loss)

# train
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

data = dataIterator(x=i, y=z)

for epoch in range(epochs):
    x_batch, y_batch = data.next_batch(bs)
    l, _ = sess.run([loss, optim], feed_dict={x: x_batch, y: y_batch})
    print("the loss is {}".format(l))

print("the weight finally is: \n", sess.run(w))
zp = sess.run(out, feed_dict={x: i})

import matplotlib.pyplot as plt

plt.plot(xs, z, "*r-")
plt.plot(xs, zp, ".-b")
plt.legend(["truth", "prediction"])
plt.show()






