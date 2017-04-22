# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

a = tf.get_variable(name = 'a',
					shape=[3,4],
					dtype = tf.float32,
					initializer=tf.random_uniform_initializer(minval=-1,maxval=1))

b = tf.argmax(input = a,dimension = 0) # 选出每列中最大值的位置
c = tf.argmax(input = a,dimension = 1) # 选出每行中最大值的位置

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(sess.run(a))



