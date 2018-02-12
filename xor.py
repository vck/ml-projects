import numpy as np

# make random stable

np.random.seed(1)

"""
A | B | A xor B
---------------
0 | 0 | 0
0 | 1 | 1
1 | 0 | 1
1 | 1 | 0

"""

"""
the task is to build a neural network that can approximate XOR function
"""

# training data
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


# set initial weight
   
w0 = np.random.randn(2,100)
w1 = np.random.randn(100, 1)

ephocs = 500
learning_rate = 0.0001

# start training

for i in range(ephocs):
   h = x.dot(w0)
   h_relu = np.maximum(h, 0)
   y_pred = h_relu.dot(w1)


   # loss
   loss = np.square(y-y_pred).sum()
   
   grad_y_pred = 2.0 * (y_pred - y)

   #  gradient of loss wrt w output layer
   grad_w1 = h_relu.T.dot(grad_y_pred)
   
   grad_h_relu = grad_y_pred.dot(w1.T)
   grad_h = grad_h_relu.copy()
   
   grad_h[h < 0] = 0
   grad_w0 = x.T.dot(grad_h)

   w0 -= learning_rate * grad_w0
   w1 -= learning_rate * grad_w1

   if i%100 == 0:
      print(i, loss)
      print(y_pred)
      print()

print(y_pred)
print(y)

print('prediction')
x = np.array([0, 1])
h = x.dot(w0)
h_relu = np.maximum(h, 0)
y_pred = h_relu.dot(w1)

print(x)
print(y_pred)








