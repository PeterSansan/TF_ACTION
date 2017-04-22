# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import time

begin=time.time()
#when put here the "cpu" is same as "gpu" , because it has been deploied on gpu or cpu
#select the fastest device automatically 
#matrix1 = np.random.rand(20000,1500).astype(np.float32)
#matrix2 = np.random.rand(1500,20000).astype(np.float32)
#product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess3:
	with tf.device("/gpu:0"):#gpu 11.6s and cpu 20.2s
		matrix1 = np.random.rand(10000,1500).astype(np.float32)
		matrix2 = np.random.rand(1500,10000).astype(np.float32)
		product = tf.matmul(matrix1, matrix2)
		result = sess3.run(product)

end = time.time()
print("Spend time %f s" %(end - begin))