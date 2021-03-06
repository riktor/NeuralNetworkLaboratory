#!/usr/bin/env python
#coding:utf-8
"""
this NN has 3 layers(input,hidden,output)
its learning way is BackPropagation


"""
import sys
import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def sign(x):
    return (np.sign(x - 0.5) + 1) / 2

class AutoEncoder(object):
    def __init__(self, n_input=None, n_hidden=None, n_output=None,
                 low=-1., high=1., alpha=0.25):
        if n_input==None or n_hidden==None or n_output==None:
            print u"please set number of each neuron"
            sys.exit()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.alpha = alpha
        
        np_rng = np.random.RandomState(123)
        self.w_i2h = np.array(np_rng.uniform(low=low,high=high,size=(self.n_hidden,self.n_input)))
        self.b_h = np.array(np_rng.uniform(low=low,high=0,size=self.n_hidden))
        self.w_h2o = np.array(np_rng.uniform(low=low,high=high,size=(self.n_output,self.n_hidden)))
        self.b_o = np.array(np_rng.uniform(low=low,high=0,size=self.n_output))

    def forward_propagation(self, input):
        if len(input) != self.n_input:
            print u"please set input-data whose size matches to n_input"
            sys.exit()
        self.input = input
        self.hidden = sigmoid(np.dot(self.w_i2h, self.input) + self.b_h)
        self.output = sigmoid(np.dot(self.w_h2o, self.hidden) + self.b_o)
        
        return self.output
        

    def back_propagation(self, teacher):
        if len(teacher) != self.n_output:
            print u"please set teacher-data whose size matches to n_output"
            sys.exit()

        oldw_h2o = self.w_h2o
        
        delta_h2o = (teacher-self.output) * self.output * (1.-self.output)
        e1 = np.array(np.mat(delta_h2o).T * np.mat(self.hidden))
        self.w_h2o = self.w_h2o + self.alpha * e1

        e2 = delta_h2o
        self.b_o = self.b_o + self.alpha * e2

        #delta_i2h = self.hidden * (1.-self.hidden) * np.dot(self.w_h2o.T , delta_h2o)
        delta_i2h = self.hidden * (1.-self.hidden) * np.dot(oldw_h2o.T , delta_h2o)
        
        f1 = np.array(np.mat(delta_i2h).T * np.mat(self.input))
        self.w_i2h = self.w_i2h + self.alpha * f1
        
        f2 = delta_i2h
        self.b_h = (self.b_h + self.alpha * f2)
        
    def train(self, data):
        self.forward_propagation(data)
        self.back_propagation(data)
        
    def test(self,data):
        return self.forward_propagation(data)

def train_ae(epoch=12000):
    nn = AutoEncoder(n_input=4, n_hidden=3, n_output=4)
    data = [np.array(map(int, format(i, 'b').zfill(4)))
            for i in xrange(16)]
    for i in xrange(epoch):
        for d in data:
            nn.train(d)
    return nn

def test_ae(nn):
    data = [np.array(map(int, format(i, 'b').zfill(4)))
            for i in xrange(16)]
    for d in data:
        out = nn.forward_propagation(d)
        print out
        print sign(out)
    
if __name__ == "__main__":
    nn = train_ae()
    test_ae(nn)

        
