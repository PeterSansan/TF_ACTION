# -*- coding:utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import time
mnist = input_data.read_data_sets('./../mnist/MNIST_data', one_hot=True)

import sys
import tensorflow as tf

f=open('cnn.txt','a+')
old=sys.stdout #将当前系统输出储存到一个临时变量中
sys.stdout=f  #输出重定向到文件
print("------------此程序运行时间-------------：%s" %time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
  
	
sess = tf.InteractiveSession()

#定义一些函数：分配系数函数、分配偏置函数、卷积函数、pooling函数
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)#均值0标准方差0.1，剔除2倍标准方差之外的随机数据
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)#统一值0.1
  return tf.Variable(initial)

def conv2d(x, W):
  #待操作的数据x，模板W，tensor不同维度上的步长，强制与原tensor等大
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    #平面数据的pool模板2*2，平面数据滑动步长2*2（非重叠的pool）
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


#x是输入的图像，y_是对应的标签
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

#第1层卷积层，Receptive Field 5＊5，单个batch生成32通道数据
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#把图像向量还原成28＊28的图像
x_image = tf.reshape(x, [-1,28,28,1])

#第1个卷积层，使用了ReLU激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第2层卷积层，Receptive Field 5＊5，单个batch 32通道生成64通道数据
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#第2个卷积层
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全链接层系数
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

#全链接层：把64通道数据展开方便全链接
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#全链层神经元使用dropout防止过拟合
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#softmax层系数
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#softmax层,在这里我试着不加softmax层，但还保留着的softmax的损失函数，也是可以的
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#交叉熵和训练构型：AdamOptimizer适合这种求和的误差项
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#验证步骤的构型
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#生成saver
saver = tf.train.Saver()

#初始化
sess.run(tf.global_variables_initializer())

#开始训练
for i in range(100000):
  batch = mnist.train.next_batch(100)
  if i%100 == 0:
    #验证的时候dropout=1.0，训练时=0.5
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
    save_path=saver.save(sess,"cnn.ckpt")
    #print save_path
    acc_score = 0.0
    for i in range(100):
      xx = mnist.test.images[i*100:(i+1)*100]
      yy = mnist.test.labels[i*100:(i+1)*100]
      acc_score = acc_score+100*accuracy.eval(feed_dict={x: xx, y_: yy, keep_prob: 1.0})
    print("当前时间：%s" %time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    sys.stdout.flush() # 刷新文件流
    print acc_score/10000

  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) # 这才是重点

#验证最终的准确率
acc_score = 0.0
for i in range(100):
  xx = mnist.test.images[i*100:(i+1)*100]
  yy = mnist.test.labels[i*100:(i+1)*100]
  # print(aa.shape,bb.shape)
  acc_score = acc_score+100*accuracy.eval(feed_dict={x: xx, y_: yy, keep_prob: 1.0})
  #print i , acc_score/((i+1)*100)
print acc_score/10000
#print "test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})


sys.stdout=old #还原原系统输出
f.close()
