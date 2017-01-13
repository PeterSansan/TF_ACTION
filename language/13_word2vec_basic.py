# -*- coding:utf-8 -*-

# 这个文档修改于word2vec_basic.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'

# 判断是否要下载文件
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)


# 把文件中的数据存为一个list，一共17005207个单词
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

words = read_data(filename)
print('Data size', len(words))
print(type(words))


# Step 2: Build the dictionary and replace rare words with UNK token.
# 字典的构造，采用的是词频最大的前50000个单词
# count：[list] 前50000个词和词频，第一个是UNK
# dictionary：[dict] 键是英文单词，值是0-50000的索引
# reverse_dictionary：[dict]键是索引，值是英文单词
# data:[list] 127005207个单词在字典中的索引值
vocabulary_size = 50000

def build_dataset(words):
  count = [['UNK', -1]]   # count是前50000个词和词频，第一个是UNK
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1)) 
  dictionary = dict()
  for word, _ in count:	
    dictionary[word] = len(dictionary) # dictionary 键是英文单词，值是0-50000的索引
  data = list()
  unk_count = 0
  for word in words: # 循环了17005207次
    if word in dictionary:
      index = dictionary[word] 
    else:
      index = 0  # dictionary['UNK'] # 如果不在50000个单词内的都被归为UNK
      unk_count += 1
    data.append(index) # data是127005207单词在字典中的索引值
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) # 键是索引，值是英文单词
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[-5:])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
print(len(data))
data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
# batch是整型的，表示上下文单词
# label也是整型的，表示目标单词
# 输出的结果格式如下：
# 如：batch_size = 6,num_skips=2,skip_window=1
# 12 as -> 3084 originated
# 12 as -> 6 a
# 6 a -> 195 term
# 6 a -> 12 as
# 195 term -> 2 of
# 195 term -> 6 a
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)  ## 定义一个双向队列
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels
  
batch, labels = generate_batch(batch_size=12, num_skips=2, skip_window=2) # 尝试输入转换结果
for i in range(12):
  print(data[i],
        '->', reverse_dictionary[data[i]])
print()
for i in range(12):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
		
# Step 4: Build and train a skip-gram model.

batch_size = 256
embedding_size = 256  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.  # 选择16个单词，计算相近值
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  # 16个验证单词，只从前面100个里面选
# 负采样相关
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])  # batch_size个数的输入
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1]) # 输入对应输出,
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)   # 验证数据集,类似这样[34，56，99，4，...]长度16

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
	# 定义词向量查询表，均匀分布初始化
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)) # 表的大小为[50000,256]
    embed = tf.nn.embedding_lookup(embeddings, train_inputs) # 输出一个[batch_size,embeddings]大小的矩阵,即[8,256]

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # NCE 相关的，这部分我还不是很懂
  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  # 求验证集与词向量表的余弦相似度
  # 构造验证步骤
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))# 原本的公式应该是对对A、B都归一化，这里略于了值小的B，即valid_dataset 
  normalized_embeddings = embeddings / norm  # 归一化后的词向量表
  valid_embeddings = tf.nn.embedding_lookup( # 输出维数[16,256]
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)  # 输出维数为[16,50000],表示的是每

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 4000001
with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  average_loss = 0
  for step in xrange(num_steps): # 每次迭代取一组batch
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels} # 这个不错

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:  # 每2000次迭代计算一次平均误差
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:  # 每10000次迭代计算一次
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval() # 这个是可视化用的，在前面5步骤中是多出来的
# Step 6: Visualize the embeddings.
# 可视化这部分我还没去看
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")