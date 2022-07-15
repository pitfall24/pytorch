import os
import torch
import math, sys, os, time
import numpy as np

class LegendrePolynomial3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        
        return 0.5 * (5 * input ** 3 - 3 * input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        
        return grad_output * 1.5 * (5 * input ** 2 - 1)

iterations = 2000

dtype = torch.float
device = torch.device('cpu')

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

learning_rate = 5e-6
for t in range(iterations):
    P3 = LegendrePolynomial3.apply
    
    y_pred = a + b * P3(c + d * x)
    
    loss = (y_pred - y).pow(2).sum()
    if t % (iterations // 20) == 0 or t == iterations - 1:
        print(t, loss.item())
    
    loss.backward()
    
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad
        
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None
        
print(f'Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} * x)')