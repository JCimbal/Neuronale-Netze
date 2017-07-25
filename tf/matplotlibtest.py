#!usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3,3,201)
# print(a)

plt.figure(figsize=(20,10))

sig = 1.0 / (1.0 + np.exp(-x))
sigplot, = plt.plot(x, sig, 'b-', label="Sigmoid")

tanh = (1 - np.exp(-2*x))/(1 + np.exp(-2*x))
tanhplot, = plt.plot(x, tanh, 'r-', label="Tanh")

relu = np.maximum(0,x)
reluplot, = plt.plot(x, relu, 'g-', label="Relu")

ax = plt.gca()

centerx = 0
centery = 0
ax.spines['left'].set_position(('data', centerx))
ax.spines['bottom'].set_position(('data', centery))
ax.spines['right'].set_position(('data', centerx))
ax.spines['top'].set_position(('data', centery))


for axis, center in zip([ax.xaxis, ax.yaxis], [centerx, centery]):
    # Turn on minor and major gridlines and ticks
    axis.set_ticks_position('both')
    axis.grid(True, 'major', ls='solid', lw=0.5, color='gray')
    axis.grid(True, 'minor', ls='solid', lw=0.1, color='gray')
    axis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    # Hide the ticklabels at <centerx, centery>
    # formatter = CenteredFormatter()
    # formatter.center = center
    # axis.set_major_formatter(formatter)

plt.legend(handles=[sigplot, tanhplot, reluplot])
# plt.legend([line_up, line_down], ['Line Up', 'Line Down'])

plt.show()
