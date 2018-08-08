import tensorflow as tf
import numpy as np

bs = 20
epochs = 1000

# dataset creation
x = np.random.random((100, 1))
y = np.random.random((100, 1))
z = 0.2 * x + 0.3 * y**2
t = np.random.random((100, 6))
i = np.concatenate([x, y, t], axis=1)


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
x = tf.placeholder(shape=[bs, i.shape[1]], dtype=tf.float32)
y = tf.placeholder(shape=[bs, 1], dtype=tf.float32)
w = tf.get_variable("w", shape=[i.shape[1], 1], initializer=tf.variance_scaling_initializer)
b = tf.get_variable("b", shape=[], initializer=tf.constant_initializer(0))

out = tf.matmul(x, w) + b

loss = tf.reduce_mean(tf.square(y-out))

# build optimizer
optim = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# train
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

data = dataIterator(x=i, y=z)

for epoch in range(epochs):
    x_batch, y_batch = data.next_batch(bs)
    l, _ = sess.run([loss, optim], feed_dict={x: x_batch, y: y_batch})
    print("the loss is {}".format(l))

print("the weight finally is: \n", sess.run(w))






