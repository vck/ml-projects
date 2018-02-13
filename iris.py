from neural import Neural
from sklearn.datasets import load_iris

net = Neural()

data = load_iris()

x, y = data.data, data.target

net.setparam(n_input=4, n_hidden=41, n_output=1, lr=0.000001)
net.init_weight()	
net.train(x, y.reshape(y.shape[0], 1), 20000)
net.accuracy()
