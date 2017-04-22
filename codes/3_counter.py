# -*- coding:utf-8 -*-

import tensorflow as tf

#design the graph
state = tf.Variable(0, name="counter")

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

#initialization
init_op = tf.global_variables_initializer()

#run 
with tf.Session() as sess:
	sess.run(init_op)
	print sess.run(state)

	for _ in range(3):
		sess.run(update)
		print sess.run(state)