# -*- coding:utf-8 -*-

import tensorflow as tf

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)		# 定义三个常量

intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
	result = sess.run([mul, intermed])
	print result
	result = sess.run([intermed])
	print result