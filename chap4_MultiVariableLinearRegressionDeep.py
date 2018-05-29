import tensorflow as tf
import numpy as np
import time

startTime = time.time()

x_data = np.array(
    [[0,0], [1,0], [1,1], [0,0], [0,0], [0,1]]
)
y_data = np.array(
    [[1,0,0],
     [0,1,0],
     [0,0,1],
     [1,0,0],
     [1,0,0],
     [0,0,1],
    ]
)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W1 = tf.Variable(tf.random_uniform([2,100],-1., 1.)) ## [특징, 은닉층의 뉴런 수]
b1 = tf.Variable(tf.zeros([100]))                    ## [은닉층의 뉴런 수]

##  신경망 추가하기
W2 = tf.Variable(tf.random_uniform([100,3],-1., 1.)) ## [은닉층의 뉴런 수, 분류 수]
b2 = tf.Variable(tf.zeros([3]))                     ## [분류 수]

L1 = tf.add(tf.matmul(X,W1), b1)
L1 = tf.nn.relu(L1)

model = tf.add(tf.matmul(L1,W2),b2)
## cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model), axis=1))
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model)
)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100) :
    sess.run(train_op,feed_dict={X:x_data, Y:y_data})
    if (step+1) % 10 == 0 :
        print("step %3d" %(step+1),sess.run(cost,feed_dict={X:x_data, Y:y_data}))

prediction = tf.argmax(model,axis=1)
target = tf.argmax(Y,axis=1)
print("예측값 : ", sess.run(prediction,feed_dict={X:x_data, Y:y_data}))
print("실제값 : ", sess.run(target,feed_dict={X:x_data, Y:y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' % sess.run(accuracy*100,feed_dict={X:x_data, Y:y_data}))

endTime = time.time() - startTime
print(endTime)