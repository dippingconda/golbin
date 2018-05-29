import tensorflow as tf
import numpy as np

# [털, 날개]
x_data = np.array(
    [[0,0], [1,0], [1,1], [0,0], [0,0], [0,1]]
)
# [기타, 포유류, 조류]
y_data = np.array(
    [[1,0,0],
    [0,1,0],
    [0,0,1],
    [1,0,0],
    [1,0,0],
    [0,0,1]]
)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_uniform([2,3],-1., 1.))   # W[m,n] = W[특징 수(털,날개) x 레이블 수(기타,포유류,조류)]
b = tf.Variable(tf.zeros([3]))                        # b[k] = b[레이블 수(기타,포유류,조류)]

# 활성화 함수 생성
L = tf.add(tf.matmul(X,W),b)
L = tf.nn.relu(L)

model = tf.nn.softmax(L)    #배열 내의 결과값들의 전체 합이 1이 되도록 만듦 = 각 결과의 확률값

# cost function 생성 (one hot encoding -> cross-entropy 함수 사용)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model), axis=1))

###################### 학습 구현 #####################
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

## init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(tf.global_variables_initializer())     # init변수를 없애서 코드를 한 줄이라도 줄이기....

## 위의 특징(x_data), 레이블(y_data) 이용하여 학습 100번 진행
for step in range(100) :
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})
    if (step+1) % 10 == 0 :
        print("learning step = ", step+1,"cost = ", sess.run(cost,feed_dict={X:x_data, Y:y_data}))

## 학습 결과 확인
# argmax : 배열 인덱스 중 가장 큰 값을 찾아줌
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값', sess.run(prediction,feed_dict={X:x_data, Y:y_data}))
print('실제값', sess.run(target,feed_dict={X:x_data, Y:y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' % sess.run(accuracy*100,feed_dict={X:x_data, Y:y_data}))
