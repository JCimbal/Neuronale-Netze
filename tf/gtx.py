#!/usr/bin/env python
#

import numpy as np
import pandas as pd
import tensorflow as tf
import math
import os
# suppress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def evaluate(value):
    dataX = [[ value ]]
    y2 = session.run([Y], feed_dict={X: dataX})
    print("data = {0}\ty2 = {1}\ty3 = {2}".format(dataX, y2, round(y2[0][0][0])))

# d = pd.DataFrame(c)
# print(d)

# d[1] = (d % 7 == 0)

samples = 10
# a = np.random.randint(10, size=samples)
a = np.linspace(0,9,samples)
print(a)

b = 1*(a<8)
print(b)

data = pd.DataFrame(data=[a,b]).T
# print(data)

n = 1
batch_size = 1
input_size = 1

# x = tf.placeholder(tf.float32, shape=(input_size))
# y = tf.add(x, x)
X = tf.placeholder(tf.float32, shape=(1,1))
W = tf.Variable(tf.zeros([1,1]))
# W = tf.Variable(tf.fill([1,1], 0.5))
b = tf.Variable(tf.zeros([1]))
Y_ = tf.placeholder(tf.float32, shape=(1,1))
# Y = tf.nn.softmax(tf.matmul(X, W) + b)
Y = tf.matmul(X, W) + b

# cross_entropy = -Y_ * tf.log(Y)
# cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y))
# lms = tf.pow(Y-Y_,2)
error = tf.square(Y - Y_)
# correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
correct_prediction = tf.equal(tf.round(Y), Y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
learningRate = 0.01
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(error)

#data = np.random.rand(input_size)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

maxEpochs = 20

for epoch in range(0, maxEpochs):

    accuracy_all = 0
    learningRate /= 2

    for i in range (0, samples):

        dataX = [[data[0][i]]]
        dataY = [[data[1][i]]]

        # print(session.run(Y, feed_dict={X: data}))
    #     a, c, W2, b2 = session.run([accuracy, cross_entropy,W,b], feed_dict={X: dataX, Y_: dataY})
    #    y, a, lms, W2, b2 = session.run([Y, accuracy, lms, W, b], feed_dict={X: dataX, Y_: dataY})
    #    y2, W2, b2, lms2, c = session.run([Y, W, b, lms, cross_entropy], feed_dict={X: dataX, Y_: dataY})
        y2, W2, b2, error2, a2 = session.run([Y, W, b, error, accuracy], feed_dict={X: dataX, Y_: dataY})

    #    print("y " + str(y) + "\ta " + str(a) + "\tlms " + str(lms))
    #    print("y " + str(y))
    #    print("x " + str(dataX) + "\t" + str(dataY) + "\ty " + str(y2) + "\tw " + str(W2) + "\tb " + str(b2) + "\tlms " + str(lms2) + "\ta " + str(a))
    #        print(str(dataX) + "\t* " + str(W2) + "\t+ " + str(b2) + "\t= " + str(y2) + "\t" + str(round(y2[0][0])) + "\t/ "+ str(dataY) + "\tlms " + str(lms2) + "\ta " + str(a))
    #        print("in: " + str(dataX) + "\t" + str(dataY) + "\ty " + str(y2) + "\tw " + str(W2) + "\tb " + str(b2) + "\tlms " + str(lms2))
        print( "{0:.3f}\t* {1:.3f}\t+ {2:.3f}\t= {3:.3f}\t=> {4}\t: {5}\te = {6:.3f}\ta = {7}".format(W2[0][0], dataX[0][0], b2[0], y2[0][0], round(y2[0][0]), dataY[0][0], error2[0][0], a2))

        accuracy_all += a2

#        if a2 == 0:
        session.run(train_step, feed_dict={X: dataX, Y_: dataY})
    #    print("w " + str(W) + "\tb " + str(b))

    print("epoch: {0}\tacc: {1}".format(epoch, accuracy_all))

k = np.linspace(0, 9)
l = evaluate(7)

# dataX = [[5]]
# y2 = session.run([Y], feed_dict={X: dataX})
# print(str(round(y2[0][0][0])))
# print(str(dataX) + "\t " + str(y2) + "\t " + str(round(y2[0][0][0])))

# session.close()

# print(str(a))
# print(str(c))
# print(str(W))
# print(str(b))

# exit

# X = tf.placeholder(tf.float32, shape=1)
# W = tf.Variable(tf.zeros([1]))
# b = tf.Variable(tf.zeros([1]))
# Y_ = tf.placeholder(tf.float32, shape=[1])
# # Y = tf.nn.softmax(tf.matmul(X, W) + b)
# # Y = tf.nn.softmax(tf.matmul(X, W) + b)
# Y = tf.add(X, X)
#
# cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y))
# correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
#
# init = tf.global_variables_initializer()
# session = tf.Session()
# sess.run(init)
#
# a, c, im, w, b = session.run([accuracy, cross_entropy], feed_dict={X: c[0][0], Y_: c[1][0]})
#
# # session.run(train_step)
#
# print(W)
# print(b)
#
# # session = tf.Session()
# # print(session.run(Y_))
