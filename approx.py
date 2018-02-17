from neural import Neural
from sklearn.datasets import load_iris
import numpy as np 
import matplotlib.pyplot as plt

import sys

server = None

net = Neural()

x = np.linspace(0, 4, 1000)
y = np.sin(x)

x_test = np.linspace(4, 8, 100)
x_test = y.reshape(y.shape[0], 1)

x = y.reshape(y.shape[0], 1)
y = x.reshape(x.shape[0], 1)


net.setparam(n_input=1, n_hidden=10, n_output=1, lr=0.00001)
net.init_weight()
net.train(x, y, 200)
net.accuracy()

cmd = sys.argv[1:]
if cmd:
    if cmd[0] == 'server' or 'serve':
        plt.plot(y)
        plt.plot([net.predict(i) for i in x_test])
        plt.show()
