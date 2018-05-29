import tensorflow as tf
import numpy as np

data = np.loadtxt('./data.csv', delimiter=',',
                  unpack=True, dtype='float32')

# 털, 날개, 기타, 포유류, 조류
# x_data = 0, 1
# y_data = 2, 3, 4
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])


global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model)
)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())   ## 이전에 정의한 변수들을 가져옴

################ 학습한 모델 재사용 ##################
chpt = tf.train.get_checkpoint_state('./model')
if chpt and tf.train.checkpoint_exists(chpt.model_checkpoint_path) :
    saver.restore(sess, chpt.model_checkpoint_path)
else :
    sess.run(tf.global_variables_initializer())

for step in range(2) :
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})
    print('Step: %d ' % sess.run(global_step),
          'Cost : %.3f' % sess.run(cost, feed_dict={X:x_data, Y:y_data}))

saver.save(sess, './model/dnn.chpt', global_step=global_step)

#### 예측 결과 및 정확도 ####
prediction = tf.argmax(model,1)
target = tf.argmax(Y,1)
print("예측값 : ", sess.run(prediction,feed_dict={X:x_data, Y:y_data}))
print("실제값 : ", sess.run(target,feed_dict={X:x_data, Y:y_data}))
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
print('정확도 : %.2f' % sess.run(accuracy*100, feed_dict={X:x_data, Y:y_data}))