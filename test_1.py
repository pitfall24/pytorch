import os
import torch
import math, sys, os, time
import numpy as np
import matplotlib.pyplot as plt

iterations = 20000

x = np.linspace(-math.pi / 2, math.pi / 2, 1000)
y = np.sin(x)

a = np.random.randn()
b = np.random.randn()
c = np.random.randn()

learning_rate = 1e-6
for t in range(iterations):
    y_pred = a / (1 + np.exp(b * x)) + c
    
    loss = np.power((y_pred - y), 2).sum()
    if t % (iterations // 20) == 0 or t == iterations - 1:
        print(t, loss)
        
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = -1 * (grad_y_pred * x).sum()
    grad_b = grad_y_pred.sum()
    grad_c = grad_y_pred.sum()
    
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    
print(f'Result: y = {a} / (1 + e^{b}x) + {c}')
print(a)
print(b)
print(c)
