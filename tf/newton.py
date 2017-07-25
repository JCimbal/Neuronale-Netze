#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def f(x):
#    return 0.5*x**2+2*x-1
#    return x**3 - 2*x + 2
#    return 0.5*x**2 + 3*x - 1
#    return pow(x,1.0/3.0)
    return (x**(1.0/3.0)).real

def df(x):
#    return x+2
#    return 3*x**2 - 2
#    return x + 3
#    return (1.0/3.0)*pow(x,-2.0/3.0)
    return ((1.0/3.0)*x**(-2.0/3.0)).real

def newton(xn):
    xnp1 = xn - f(xn)/df(xn)
    return xnp1

def plot(m,b):

# def main():
maxIterations = 10
x0 = 1
xnp1 = x0
eps = 0.00001

for i in range(1, maxIterations):
    xn = xnp1
    xnp1 = newton(xn)

    print("xn = {0:.3f}\tf(xn) = {1:.3f}\tf'(xn) = {2:.3f}\txnp1 = {3:.3f}".format(xn, f(xn), df(xn), xnp1))

    if (abs(xn-xnp1) < eps):
        print("below eps")
        break

    m = (f(xn)-0) / (xn-xnp1)
    b = -m * xnp1
    print("\tm = {0:.3f}\tb = {1:.3f}".format(m, b))

#        plot(m,b)

# if __name__ == "__main__":
#     main()
