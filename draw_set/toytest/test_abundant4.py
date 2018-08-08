import tensorflow as tf
import numpy as np

#create data
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

#create tensorflow structure
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0)) #一维，范围[-1,1]
biases=tf.Variable(tf.zeros([1]))

y=Weights*x_data+biases

loss=tf.reduce_mean(tf.square(y-y_data))

#建立优化器，减小误差，提高参数准确度，每次迭代都会优化
optimizer=tf.train.GradientDescentOptimizer(0.5) #学习效率<1
train=optimizer.minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #train
    for step in range(201):
        sess.run(train)
        if step%20==0:
            print(step,sess.run(Weights),sess.run(biases))