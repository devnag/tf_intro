#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import random
import sys

### Test linear regression
coefficients = [3.14159, 2.71828, 0.1234567]
numVars = len(coefficients)
numSamples = 1000
noiseLevel = 1e-2
learning_rate = 2e-2

# Create placeholders for input variables
input_placeholder = tf.placeholder(tf.float32, shape=(numVars, 1))
output_placeholder = tf.placeholder(tf.float32, shape=(1,1))

# Create the only variable -- vector of weights
weights = tf.Variable(tf.zeros([1, numVars]), name="weights")

# Finish building graph; loss is squared error
dotProduct = tf.matmul(weights, input_placeholder) 
loss = tf.square(dotProduct - output_placeholder)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Create random data, and add noise.
in_data = np.random.rand(numSamples, numVars)
out_data = np.dot(in_data, np.array(coefficients)) + (noiseLevel * np.random.rand(numSamples))

for step in range(20000):
  randIndex = random.randint(0, numSamples-1)
  results = sess.run([train,weights,dotProduct], feed_dict = 
                {input_placeholder: in_data[randIndex].reshape(numVars, 1), output_placeholder: out_data[randIndex].reshape(1,1)})
  if ((step % 100) == 0):
    print("Step %s" % (step))
    print("%s => %s " % (results[1], results[2] - out_data[randIndex]))
  
print(coefficients, " were the actual coefficients.")
