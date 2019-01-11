import tensorflow as tf
from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# with tf.device(':gpu:0/'):
#     x = tf.Variable(3, tf.float32, name='x')
# y = tf.Variable(4, tf.float32, name='y')
# W = x**2*y+x*y
#
# init = tf.global_variables_initializer()
#
# sess = tf.InteractiveSession()
# init.run()
# result = W.eval()
#
# print(result)
# sess.close()

# x1 = tf.Variable(1)
# print(x1.graph is tf.get_default_graph())
#
# graph = tf.Graph()
# with graph.as_default():
#     x2 = tf.Variable(2)
#
# print(x2.graph is tf.get_default_graph())
# print(x2.graph is graph)

# w = tf.Variable(5)
# x = w + 2
# y = x + 5
# z = x * 3
#
# #正常写法
# with tf.Session() as sess:
#     sess.run(w.initializer)
#     print(sess.run(y))
#     print(sess.run(z))
#
# #技巧
# with tf.Session() as sess:
#     sess.run(w.initializer)
#     y_val, z_val = sess.run([y, z])
#     print(y_val, z_val)
# housing = fetch_california_housing()
# m, n = housing.data.shape
# housing_plus_bias = np.c_[np.ones((m, 1)), housing.data]
#
# X = tf.constant(housing_plus_bias, dtype=tf.float32, name='X')
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
# XT = tf.transpose(X)
# theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
# with tf.Session() as sess:
#     result = theta.eval()
#
# print(result)

# #初始化循环次数
# n_epochs = 36500
# #定义学习率
# learning_rate = 0.001
#
# #获取数据集
# housing = fetch_california_housing(download_if_missing=True)
# #获取数据集的维度
# m, n = housing.data.shape
# #给数据集添加偏置
# housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]
# #拟合数据 准备归一化 默认方差归一化和均值归一化
# scaler = StandardScaler().fit(housing_data_plus_bias)
# #获取归一化后的数据
# scaler_housing_data_plus_bias = scaler.transform(housing_data_plus_bias)
# #设置X节点
# X = tf.constant(scaler_housing_data_plus_bias, dtype=tf.float32, name='X')
# #设置y节点
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
# #设置theta的初始值 使用random_uniform
# theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
# #预测y_hat
# y_pre = tf.matmul(X, theta, name='y_pre')
# #计算误差
# error = y_pre - y
# #计算rmes
# rmse = tf.sqrt(tf.reduce_mean(tf.square(error), name='rmse'))
# #根据公式 2/m*(XT*error)计算梯度
# # gradients = 2/m * tf.matmul(tf.transpose(X), error)
# gradients = tf.gradients(rmse, [theta])[0]
# #更新theta
# training_op = tf.assign(theta, theta - learning_rate * gradients)
# #全局初始化
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     for epoch in range(n_epochs):
#         if epoch % 100 == 0:
#             print(epoch, rmse.eval())
#         sess.run([training_op])
#
#     best_theta = theta.eval()
#
#     print(best_theta)
a = time.time()
n_epochs = 1000
learning_rate = 0.001
batch_size = 2000

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

scaler = StandardScaler().fit(housing_data_plus_bias)
scaler_housing_data_plus_bias = scaler.transform(housing_data_plus_bias)

X_train = scaler_housing_data_plus_bias[:18000]
X_test = scaler_housing_data_plus_bias[18000:]
y_train = housing.target.reshape(-1, 1)[:18000]
y_test = housing.target.reshape(-1, 1)[18000:]

X = tf.placeholder(dtype=tf.float32, name='X')
y = tf.placeholder(dtype=tf.float32, name='y')

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
rmse = tf.sqrt(tf.reduce_mean(tf.square(error)), name='error')

traininge_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(rmse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    n_batch = int(18000 / batch_size)

    for epoch in range(n_epochs):
        print('Epoch:', epoch, 'train_RMSE:',
              sess.run(rmse, feed_dict={X: X_train,
                                        y: y_train
                                        }
                       ))
        print('Epoch:', epoch, 'test_RMSE:',
              sess.run(rmse, feed_dict={X: X_test,
                                        y: y_test
                                        }
                       ))
        arr = np.arange(18000)
        np.random.shuffle(arr)
        X_train = X_train[arr]
        y_train = y_train[arr]

        for i in range(n_batch):
            sess.run(traininge_op, feed_dict={
                X: X_train[i*batch_size: i*batch_size + batch_size],
                y: y_train[i*batch_size: i*batch_size + batch_size]
            })

    best_theta = theta.eval()
    print(best_theta)
b = time.time()
print(b-a)