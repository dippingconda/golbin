import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

## hyper parameters
learning_rate = 0.01
training_epoch = 20
batch_size = 100
n_hidden = 256
n_input = 28*28

X = tf.placeholder(tf.float32, [None, n_input])

## encoder
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))

## decoder
W_decoder = tf.Variable(tf.random_normal([n_hidden,n_input]))
b_decoder = tf.Variable(tf.random_normal([n_input]))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder,W_decoder),b_decoder))

## cost fuction
cost = tf.reduce_mean(tf.pow(X-decoder, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

## 학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(training_epoch) :
    total_cost = 0
    for i in range(total_batch) :
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X:batch_xs})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch+1),
          'Avg. cost =', '{:.4f}'.format(total_cost/total_batch))

print('최적화 완료!')

## 결과 확인
sample_size = 10
samples = sess.run(decoder,feed_dict={X:mnist.test.images[:sample_size]})

fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

for i in range(sample_size) :
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i],(28, 28)))
    ax[1][i].imshow(np.reshape(samples[i],(28, 28)))

plt.show()
