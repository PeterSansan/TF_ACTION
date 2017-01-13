# -*- coding:utf-8 -*-

#download the MNIST data in folder "MNIST_data" that in the same path as this *.py
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

#图片的占位
x = tf.placeholder(tf.float32, [None, 784])

#系数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#softmax层
y = tf.matmul(x,W)+b
#y = tf.nn.softmax(tf.matmul(x, W) + b)

#用于训练的真实值占位
y_ = tf.placeholder(tf.float32, [None, 10])

#交叉熵：-tf.reduce_sum(y_ * tf.log(y)是一个样本的，外面的tf.reduce_mean是batch的
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
#规定训练的方法：注意：使用GradientDescentOptimizer适合上述的误差项
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#初始化
init = tf.initialize_all_variables()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True #限制GPU


sess = tf.Session(config=config)
sess.run(init)

#训练
# for i in range(10000):
  # batch_xs, batch_ys = mnist.train.next_batch(100)
  # #print batch_xs.shape
  # sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # #验证，argmax(y,1)是获得y的第一个维度（即每一行）的最大值的位置
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(i,sess.run([accuracy,tf.shape(y)], feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


