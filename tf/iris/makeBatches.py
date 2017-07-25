#!/usr/bin/env python

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import urllib

import tensorflow as tf
import numpy as np

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
IRIS_TEST = "iris_test.csv"

IRIS_COMBINED = "iris_combined.csv"

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

combined_set = training_set

with open(IRIS_TRAINING,'r') as f:
    traininglines = f.read().split("\n")

with open(IRIS_TEST,'r') as f:
    testlines = f.read().split("\n")

# traininglines[1:-1]

# testlines[1:-1]

datalines = traininglines[1:-1] + testlines[1:-1]
# datalines

header = traininglines[0].split(",")
# header
header[0] = str(len(datalines))
# header
header2 = ",".join(header)
# header2

datalines.insert(0, header2)

datalines

with open(IRIS_COMBINED,'w') as f:
    for line in datalines:
        f.write(line + "\n")
