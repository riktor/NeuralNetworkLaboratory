# coding: utf-8
# online learning
                                  
import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def sign(x):
    return (np.sign(x - 0.5) + 1) / 2

def grad(activation, x):
    if activation == sigmoid:
        return activation(x) * (1 - activation(x))
    if activation == tanh:
        return 1 - activation(x) * activation(x)

"""
# kouiuno mo ari
H = 10e-10
def grad(f):
    def f_prime(x):
        Hv = np.array([H for i in xrange(len(x))])
        return (f(x+Hv) - f(x))/H
    return f_prime
"""

def sign(x):
    return (np.sign(x - 0.5) + 1) / 2

class Network(object):
    def __init__(self, layer_size=None): 
        if layer_size is None:
            raise Exception(u"error: set layer_size")
        self.layers = []
        np_rng = np.random.RandomState(1234)
        for i in xrange(len(layer_size) - 1):
            layer = Layer(size=layer_size[i], next_size=layer_size[i+1], np_rng=np_rng)
            self.layers.append(layer)

    def forward_propagation(self, input=None):
        if input is None:
            raise Exception(u"error: set input")
        output = input
        for layer in self.layers:
            output = layer.get_output(output)
        return output

    def back_propagation(self, target=None):
        if target is None:
            raise Error(u"error: set target")
        backlayers = self.layers[::-1]
        layer = backlayers[0]
        delta = (target - layer.output) * np.array(layer.output) * (1. - np.array(layer.output))
        #delta = (target - layer.output) * grad(layer.activation, layer.output)
        #print np.mat(delta).shape
        #print np.mat(layer.input).shape
        layer.w += layer.alpha * np.array(np.mat(layer.input).T * np.mat(delta))
        layer.b += layer.alpha * delta
        delta = np.array(layer.input) * (1. - np.array(layer.input)) * np.dot(layer.w, delta.T)
        for layer in backlayers[1:]:
            #print delta
            layer.w += layer.alpha * np.array(np.mat(layer.input).T * np.mat(delta))
            layer.b += layer.alpha * delta
            delta = np.array(layer.input) * (1. - np.array(layer.input)) * np.dot(layer.w, delta.T) 
    
    def train(self, input, target):
        self.forward_propagation(input) # print self.forward_propagation, target
        self.back_propagation(target)

class Layer(object):
    def __init__(self, size=None, next_size=None, 
                 low=-1., high=1., alpha=0.7, 
                 activation=sigmoid, np_rng=None):
        if size is None or next_size is None:
            raise Exception(u"error: set size")
        self.alpha = alpha
        self.activation = activation
        if np_rng is None:
            np_rng = np.random.RandomState(123)
        self.np_rng = np_rng
        self.w = np.array(np_rng.uniform(low = low, 
                                         high = high,
                                         size = (size, next_size)))
        self.b = np.array(np_rng.uniform(low = low,
                                         high = high,
                                         size = next_size))

    def get_output(self, input):
        #print "w",self.w.T
        #print "i",input
        #print "b",self.b
        self.input = input
        self.output = self.activation(np.dot(self.w.T, input) + self.b)
        return self.output


def test(epoch=10000):
    nn = Network([2, 3, 1])
    data = np.array([[[0,0],[0]],
                     [[0,1],[1]],
                     [[1,0],[1]],
                     [[1,1],[0]]])
    
    for i in xrange(epoch):
        for d in data:
            nn.train(d[0], d[1])
    
    for d in data:
        print nn.forward_propagation(d[0]),sign(nn.forward_propagation(d[0])), d[1]

def test2(epoch=10000):
    nn = Network([3,2,3])
    data = np.array([[[0,0,0],[0,0,0]],
                     [[0,0,1],[0,0,1]],
                     [[0,1,0],[0,1,0]],
                     [[0,1,1],[0,1,1]],
                     [[1,0,0],[1,0,0]],
                     [[1,0,1],[1,0,1]],
                     [[1,1,0],[1,1,0]],
                     [[1,1,1],[1,1,1]]])
    
    for i in xrange(epoch):
        for d in data:
            nn.train(d[0], d[1])
    
    for d in data:
        print nn.forward_propagation(d[0]),sign(nn.forward_propagation(d[0])),d[1]

if __name__ == "__main__":
    test()
