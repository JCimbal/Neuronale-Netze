#!/usr/bin/env python

import os
import urllib

import tensorflow as tf
import numpy as np

IRIS_TRAINING = "iris_training_new.csv"
IRIS_TEST = "iris_test_new.csv"
IRIS_COMBINED = "iris_combined.csv"

TestRatio = 0.10

with open(IRIS_COMBINED,'r') as f:
    lines = f.read().split("\n")

Batch1 = []
Batch2 = []

for line in lines[1:-1]:
    if np.random.random() < TestRatio:
        Batch1.append(line)
    else:
        Batch2.append(line)

len(Batch1)
len(Batch2)

Batch1
Batch2

with open(IRIS_TRAINING,'w') as f:
    for line in traininglines:
        f.write(line + "\n")

with open(IRIS_TEST,'w') as f:
    for line in testlines:
        f.write(line + "\n")
