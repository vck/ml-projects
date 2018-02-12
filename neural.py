# neural network framework built on top of numpy

import numpy as np

np.random.seed(1)

class Neural(object):
   def __init__(self):
      self.n_input = None
      self.n_hidden = None
      self.n_output = None
      self.weights = None
      self.learning_rate = None

   def setparam(self, **kwargs):
      self.n_input = kwargs['n_input']
      self.n_hidden = kwargs['n_hidden']
      self.n_output = kwargs['n_output']
      self.learning_rate = kwargs['lr']

   def init_weight(self):
      self.weights = {}
      self.weights['w0'] = 0.5 * np.random.randn(self.n_input, self.n_hidden)
      self.weights['w1'] = 0.5 * np.random.randn(self.n_hidden, self.n_output)
      

   def train(self, x, y, ephocs):

      self.loss = None

      for i in range(ephocs):
         h = x.dot(self.weights['w0'])
         h_relu = np.maximum(h, 0)
         y_pred = h_relu.dot(self.weights['w1'])
         
         self.loss = np.square(y-y_pred).sum()

         # most important lines
         grad_y_pred = 2.0 * (y_pred-y)
         # update weights with backprop
         grad_w1 = h_relu.T.dot(grad_y_pred)
         grad_h_relu = grad_y_pred.dot(self.weights['w1'].T)

         grad_h = grad_h_relu.copy()

         grad_h[h < 0] = 0

         grad_w0 = x.T.dot(grad_h)
         
         self.weights['w0'] -= self.learning_rate * grad_w0
         self.weights['w1'] -= self.learning_rate * grad_w1
         
         if i%10 == 0:
            print(i, self.loss)
      
   
      
   def get_weights(self):
      return self.weights


   def predict(self, x):
      h = x.dot(self.weights['w0'])
      h_relu = np.maximum(h, 0)
      y_predict = h_relu.dot(self.weights['w1'])
      return y_predict

   def accuracy(self):
      acc = 100-self.loss
      print(acc)

if __name__ == "__main__":
   x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
   y = np.array([[0], [1], [1], [0]])

   net = Neural()
   net.setparam(n_input=2, n_hidden=300, n_output=1, lr=0.0001 )
   net.init_weight()
   #print(net.get_weights())
   
   print('training...')

   net.train(x, y, 400)

   #print(net.get_weights())
            
   print(net.predict(np.array([0, 1])))
   print(net.predict(np.array([1, 1])))
   print(net.predict(np.array([1, 0])))
   print(net.predict(np.array([1, 1])))
