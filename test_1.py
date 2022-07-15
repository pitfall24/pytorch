import os
import torch
import math, sys, os, time, matplotlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

iterations = 2000

x = np.linspace(-math.pi / 2, math.pi / 2, 1000)
y = np.sin(x)

plt.plot(x, y)
plt.show()