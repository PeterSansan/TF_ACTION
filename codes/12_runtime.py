# -*- coding: utf-8 -*-  
import tensorflow as tf  
import time  
  
t1 = time.time()  
x = tf.Variable([1.0])  
b =1.0  
start1 = time.time()  
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
    #通过sess.run的方式读变量  
    for step in range(100000):  
        res = sess.run(x)  
    print "通过sess.run的方式读变量所需的时间:",time.time()-start1  
    start2 = time.time()  
    for step in range(100000):  
        a = b  
    print "通过直接赋值的手段读变量所需的时间:",time.time()-start2  