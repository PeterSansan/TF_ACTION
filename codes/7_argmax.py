# -*- coding:utf-8 -*-

import tensorflow as tf
from pprint import pprint

a = tf.get_variable('a',shape=[5,2])    # 默认 trainable=True

b = tf.get_variable('b',shape=[2,5],trainable=False)

c = tf.constant([1,2,3],dtype=tf.int32,shape=[8],name='c') # 因为是常量，所以trainable=False

d = tf.Variable(tf.random_uniform(shape=[3,3]),name='d')

tvar = tf.trainable_variables()

tvar_name = [x.name for x in tvar]
print(tvar)
# [<tensorflow.python.ops.variables.Variable object at 0x7f9c8db8ca20>, <tensorflow.python.ops.variables.Variable object at 0x7f9c8db8c9b0>]
print(tvar_name)
# ['a:0', 'd:0']

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
pprint(sess.run(tvar))
#[array([[ 0.27307487, -0.66074866],
#       [ 0.56380701,  0.62759042],
#       [ 0.50012994,  0.42331111],
#       [ 0.29258847, -0.09185416],
#       [-0.35913971,  0.3228929 ]], dtype=float32),
# array([[ 0.85308731,  0.73948073,  0.63190091],
#       [ 0.5821209 ,  0.74533939,  0.69830012],
#       [ 0.61058474,  0.76497936,  0.10329771]], dtype=float32)]