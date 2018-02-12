# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N = 64
D_in = 1000
H = 100
D_out = 10

# D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

loss_data = []

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)
    loss_data.append(loss)
    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2



#hidden_weight = np.random.randn(1, x.shape[1])
#output_weight = np.random.randn(x.shape[1], 1)

#plt.figure()

#plt.scatter(x.argmax(axis=1), y.argmax(axis=1))
#plt.scatter(x.argmax(axis=1), y_pred.argmax(axis=1))

plt.plot([i for i in range(len(loss_data))], loss_data)
plt.show()
