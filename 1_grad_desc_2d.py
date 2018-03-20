#!/usr/bin/env python

import math
import numpy as np

# Point is a 2d vector
new_pt = lambda x,y: np.array([x,y])

loss_f = lambda p: 4 + (0.5 * math.pow(p[0] - 7.5, 2.0)) + (2.1 * math.pow(p[1] - 5.1, 3.0))

def deriv(fn, p):
    delta = 1e-2
    pderiv_x = (fn(p + new_pt(delta,0.0)) - fn(p) )/ delta
    pderiv_y = (fn(p + new_pt(0.0, delta)) - fn(p) )/ delta
    return new_pt(pderiv_x, pderiv_y)          

p0 = new_pt(10, 10)
alpha = 1e-3
p = p0

for i in range (50000):
    loss_at_p = loss_f(p)
    loss_deriv_at_p = deriv(loss_f, p)
    p = (p - (alpha * loss_deriv_at_p))
    if i % 100 == 0:
        print("%s: x = %s, Loss %s, derivative %s" % (i, p, loss_at_p, loss_deriv_at_p))




