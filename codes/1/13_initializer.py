#! /usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import tensorflow as tf

value = [0,1,2,3,4,5,6,7]
#value = np.array(value)  # 注释掉没有影响
#value = value.reshape([2,4])

init = tf.constant_initializer(value)

print('fitting shape:')
tf.reset_default_graph()
with tf.Session():
  x = tf.get_variable('x',shape=[2,4],initializer=init)
  x.initializer.run()
  print(x.eval())
  
#  fitting shape:
#  [[ 0.  1.  2.  3.]
#   [ 4.  5.  6.  7.]]

print('larger shape:')
tf.reset_default_graph()
with tf.Session():
  x = tf.get_variable('x',shape=[3,4],initializer = init)
  x.initializer.run()
  print(x.eval())

# large shape:
 # [[ 0.  1.  2.  3.]
 # [ 4.  5.  6.  7.]
 # [ 7.  7.  7.  7.]]
 
# print('smaller shape:')
# tf.reset_default_graph()
# with tf.Session():
  # x = tf.get_variable('x',shape=[2,3],initializer=init)  # 这种小型的初始化方式是不允许的

  
print('random_normal_initializer:')
tf.reset_default_graph()
init = tf.random_normal_initializer(mean=0.0,stddev=1.0)
with tf.Session():
  x = tf.get_variable('x',shape=[3,4],initializer = init)
  x.initializer.run()
  print(x.eval())
  
# random_normal_initializer:
 # [[-0.92622328  1.83604741  0.3032876  -0.18343297]
 # [-2.93835282  0.56074178  0.45900798 -0.88488728]
 # [-1.50388312  1.02886796  0.9727664  -0.63334048]]

print('truncated_normal_initializer:')
tf.reset_default_graph()
init = tf.truncated_normal_initializer(mean=0.0,stddev=1.0)
with tf.Session():
  x = tf.get_variable('x',shape=[3,4],initializer = init)
  x.initializer.run()
  print(x.eval())
  
# truncated_normal_initializer
 # [[-1.07060325  0.70100886 -0.9471904  -0.41293758]
 # [-0.71320456  0.94850993 -0.59993124  0.7818318 ]
 # [ 0.76422453 -0.41598266  0.0457042   0.33857414]]

print('random_uniform_initializer:')
tf.reset_default_graph()
init = tf.random_uniform_initializer(minval=0,maxval=None)
with tf.Session():
  x = tf.get_variable('x',shape=[3,4],initializer = init)
  x.initializer.run()
  print(x.eval())

# random_uniform_initializer:
 # [[ 0.98354113  0.10508585  0.38734615  0.48602521]
 # [ 0.40941119  0.78411257  0.88414943  0.09296131]
 # [ 0.25924373  0.96777785  0.33620894  0.40009141]]