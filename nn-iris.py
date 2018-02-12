import numpy as np
from sklearn.datasets import load_iris

data = load_iris()

x, y = data.data, data.target

# initialize weight 

w_hidden = np.random.randn(x.shape[1], 100)
w_out = np.random.randn(100, 1)

lr = 1e-6

for _ in range(500):
   
   h = x.dot(w_hidden)
   h_relu = np.maximum(h, 0)
   y_pred = h_relu.dot(w_out)

   loss = np.square(y_pred - y).sum()
   print(loss)

   grad_y_pred = 2 * (y_pred - y)
   
   grad_w = h_relu.T.dot(grad_y_pred)
   
   
   grad_h_relu = grad_y_pred.dot(w_out.T)
   #grad_h = grad_h_relu.copy()
   #grad_h[h < 0] = 0
   #grad_w1 = x.T.dot(grad_h)


   #w_hidden -= lr * grad_w1
   #w_out -= lr * grad_w2


