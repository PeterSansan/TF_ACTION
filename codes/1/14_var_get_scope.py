# -*- coding:utf-8 -*-

import tensorflow as tf

sess = tf.InteractiveSession()
# 不同的命名域与变量
with tf.variable_scope("foo"):
  with tf.variable_scope("bar"):
    v = tf.get_variable("v", [1])
    print(v.name)

print(v.name)
with tf.variable_scope("foo1"):
  with tf.variable_scope("bar1"):
    v = tf.get_variable("v", [1])
    print(v.name)
	
print(v.name)

# 变量共享（相同变量）

with tf.variable_scope("xxx"):
  a = tf.get_variable("a",[1])
with tf.variable_scope("xxx",reuse=True):	# 采用xxx/a的值
  a1 = tf.get_variable("a",[1])
print(a,a1)
sess.run(tf.global_variables_initializer())
print(a.eval(),a1.eval()) #如果初始化为None,则会采用variable_scope的初始化值，
# 如果也是None,则采用uniform_unit_scaling_initializer
assert a==a1  # a , a1 是一样的

# 变量共享（相同变量）另一种写法
with tf.variable_scope("yyy") as scope:
    v = tf.get_variable("v", [1])
    scope.reuse_variables()
    v1 = tf.get_variable("v", [1])
assert v1 == v

# 为防止在没有使用reuse的情况下出现相现的共享变量，则会弹出异常,如下面是有错误的
# with tf.variable_scope("zzz"):
  # v = tf.get_variable("v",[1])
  # v1 = tf.get_variable("v",[1])
  # Raises ValueError("...v already exists ...").
  
# 为防止在使用reuse的情况下引用了之前没有的共享变量，则会弹出异常，如下面是有错误的
# with tf.variable_scope("aaa",reuse=True):
  # v = tf.get_variable("v",[1])
  # Raises ValueError("... v does not exists...").