# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

#deploy a session
sess = tf.InteractiveSession()

#design the grape
matrix1 = np.random.rand(2000,1500).astype(np.float32)
matrix2 = np.random.rand(1500,2000).astype(np.float32)
product = tf.matmul(matrix1, matrix2)

#run the operation
print product.eval()

sess.close()