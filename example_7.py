import os
import torch
import math, sys, os, time
import numpy as np

class Plynomial3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        
    def forward(self, x):
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
    
    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

iterations = 2000

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

model = Plynomial3()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(iterations):
    y_pred = model(x)
    
    loss = criterion(y_pred, y)
    if t % (iterations // 20) == 0 or t == iterations - 1:
        print(t, loss.item())
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
print(f'Result: {model.string()}')