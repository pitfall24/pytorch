import random
import matplotlib.pyplot as plt

iterations = 2000
numbers = 2000

x = [i for i in range(numbers)]
y = [x[i] * 0.3 + random.randint(i - numbers // 10, i + numbers // 10) for i in range(numbers)]

a = random.randint(-100, 100)
b = random.randint(-100, 100)

learning_rate = 1e-6
for t in range(iterations):
    y_pred = [a + b * i for i in x]
    
    loss = sum([abs(y[i] - y_pred[i]) for i in range(numbers)])
    if t % (iterations // 20) == 0 or t == iterations - 1:
        print(f'{t}\t{loss}\t\t{a}\t{b}')
        
    grad_y = [2 * (y_pred[i] - y[i]) for i in range(numbers)]
    grad_a = sum(grad_y)
    grad_b = sum([grad_y[i] * x[i] for i in range(numbers)])
    
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b / (2 * numbers)


print(f'y = {b}x {"+" if a >= 0 else "-"} {abs(a)}')

xf, yf = [i for i in range(numbers)], [a + b * i for i in range(numbers)]

plt.plot(x, y)
plt.plot(xf, yf)
plt.show()