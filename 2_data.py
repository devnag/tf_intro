#!/usr/bin/env python

import tensorflow as tf
import numpy as np

x = [[1.0,2.0], [3.0,4.0]]
y = np.array(x)
z = tf.constant(y)
print(type(x), type(y), type(z))
print(x, y, z)
print()

sess = tf.Session()
y_ = sess.run(z)
x_ = y_.tolist()

print(type(x_), type(y_), type(z))
print(x_, y_, z)
