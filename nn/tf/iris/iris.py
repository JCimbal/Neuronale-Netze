#!/usr/bin/env python

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import urllib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

# Define the training inputs
def get_train_inputs():
  x = tf.constant(training_set.data)
  y = tf.constant(training_set.target)
  #  y = tf.summary.scalar(, training_set.target)
  return x, y

# Define the test inputs
def get_test_inputs():
  x = tf.constant(test_set.data)
  y = tf.constant(test_set.target)
  return x, y

if not os.path.exists(IRIS_TRAINING):
  raw = urllib.urlopen(IRIS_TRAINING_URL).read()
  with open(IRIS_TRAINING,'w') as f:
    f.write(raw)

if not os.path.exists(IRIS_TEST):
  raw = urllib.urlopen(IRIS_TEST_URL).read()
  with open(IRIS_TEST,'w') as f:
    f.write(raw)

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

training_set.target[0]

colors = [000000, 111111, 222222]
colors[0]

x = [1]
y = [1]
col = []

n = 120
r = 2 * np.random.rand(n)
theta = 2 * np.pi * np.random.rand(n)
area = 200 * r**2 * np.random.rand(n)
colors = theta
colors = training_set.target
colors = np.random.rand(n)
len(colors)
len(training_set.data[:,1])
c = plt.scatter(theta, r, c=colors, s=20, cmap=plt.cm.Blues)
c = plt.scatter(training_set.data[:,0], training_set.data[:,1], c=colors, s=20, cmap=plt.cm.cool)
plt.show()

plt.plot(training_set.data[:,1], training_set.data[:, 0], ".", color=colors[int(training_set.target)])
plt.show()

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model9")

classifier.fit(input_fn=get_train_inputs, steps=3000)

accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]

print("\nTest Accuracy: {0:.1f}%\n".format(100*accuracy_score))

#########################################

# Classify two new flower samples.
def new_samples():
  return np.array(
    [[6.4, 3.2, 4.5, 1.5],
     [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

predictions = list(classifier.predict_classes(input_fn=new_samples))

print( "New Samples, Class Predictions:    {}\n".format(predictions))
