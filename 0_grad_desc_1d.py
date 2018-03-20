#!/usr/bin/env python

import math

loss_f = lambda x: 4 + math.pow(x - 3.5, 2.0)

def deriv(fn, x_val):
    delta = 1e-2
    return (fn(x_val + delta) - fn(x_val)) / delta

x0 = 1e6
alpha = 0.1
x = x0

for i in range (100):
    loss_at_x = loss_f(x)
    loss_deriv_at_x = deriv(loss_f, x)
    x = (x - (alpha * loss_deriv_at_x))
    print("%s: x = %s, Loss %s, derivative %s" % (i, x, loss_at_x, loss_deriv_at_x))




