# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

input1 = tf.placeholder(tf.float32,shape=(5, 5))
input2 = tf.placeholder(tf.float32,shape=(5, 5))
output = tf.matmul(input1, input2)#matmul is different mul

with tf.Session() as sess:
	rand_array = np.ones([5, 5])
	print sess.run([output], feed_dict={input1: rand_array,input2: rand_array})