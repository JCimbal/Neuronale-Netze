#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def softmax(v):

    v2 = np.exp(v)
    #    print(v2)

    sum = np.sum(v2)
    #    print(sum)

    v3 = v2/sum
    #    print(v3)

    return v3

def plot(o, sm):

    x = np.linspace(0,10,11)

    plt.figure(figsize=(20,10))

    #    oplot, = plt.plot(x, o, 'g-', label="x")
    smplot, = plt.plot(x, sm, 'b-', label="sm")

    ax = plt.gca()

    centerx = 0
    centery = 0
    ax.spines['left'].set_position(('data', centerx))
    ax.spines['bottom'].set_position(('data', centery))
    ax.spines['right'].set_position(('data', centerx))
    ax.spines['top'].set_position(('data', centery))

    for axis, center in zip([ax.xaxis, ax.yaxis], [centerx, centery]):
        axis.set_ticks_position('both')
        axis.grid(True, 'major', ls='solid', lw=0.5, color='gray')
        axis.grid(True, 'minor', ls='solid', lw=0.1, color='gray')
        axis.set_minor_locator(mpl.ticker.AutoMinorLocator())

        #    plt.legend(handles=[xplot, smplot])

    plt.show()

#    o = np.linspace(0,1,11) # list(range(0,1,11))
o = [0, 0, 0, 0, 9, 0, 1, 0, 2, 0, 0]
print("o = {0}".format(o))

sm = softmax(o)
print("softmax = {0}".format(sm))

plot(o, sm)
