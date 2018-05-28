import tensorflow as tf

# tensor & graph generation
hello = tf.constant('Hello, TensorFlow!')
print(hello)

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)
print(c)

sess = tf.Session()
print(sess.run(hello))
print(sess.run([a,b,c]))

# placeholder & variable
X = tf.placeholder(tf.float32,[None, 3])
print(X)
x_data = [[1,2,3], [4,5,6]]

W = tf.Variable(tf.random_normal([3, 2])) # normal distribution
b = tf.Variable(tf.random_normal([2, 1])) # normal distribution
### W = tf.Variable([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])  # explicit data
### b = tf.Variable([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])  # explicit data

expr = tf.matmul(X,W) + b

sess.run(tf.global_variables_initializer())     ## variable 초기화가 필요
print("===x_data===")
print(x_data)
print("=== W ===")
print(sess.run(W))
print("=== b ===")
print(sess.run(b))
print("=== expr ===")
print(sess.run(expr, feed_dict={X:x_data}))     ## feed_dict : dictionary type으로 placeholder에 값을 입력함

sess.close()
