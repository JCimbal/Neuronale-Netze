#!/usr/bin/env python

from sklearn.cross_validation import train_test_split
import numpy

with open("datafile.txt", "rb") as f:
   data = f.read().split('\n')
   data = numpy.array(data)  #convert array to numpy type array

   x_train ,x_test = train_test_split(data,test_size=0.5)       #test_size=0.5(whole_data)
   
