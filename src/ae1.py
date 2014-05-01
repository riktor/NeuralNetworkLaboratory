#!/usr/bin/env python
#coding: utf-8

import sys
import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))
def sign(x):
    return (np.sign(x - 0.5) + 1) / 2

class AutoEncoder(object):
    def __init__(self, n_input=None, n_output=None,
                 low=-1., high=1., alpha=0.25):
        if n_input==None or n_output==None:
            print u"please set number of each neuron"
            sys.exit()
        self.n_input = n_input
        self.n_output = n_output
        self.alpha = alpha
        
        np_rng = np.random.RandomState(123)
        self.w_i2o = np.array(np_rng.uniform(low=low,high=high,size=(self.n_output,self.n_input)))
        self.b_h = np.array(np_rng.uniform(low=low,high=high,size=self.n_output))

    def ForwardPropagation(self, input):
        #print "input",
        #print input
        if len(input) != self.n_input:
            print u"please set input-data whose size matches to n_input"
            sys.exit()
        self.input = input
        self.hidden = sigmoid(np.dot(self.w_i2o, self.input))# + self.b_h)
        self.output = sigmoid(np.dot(self.w_i2o.T, self.hidden))# + self.b_o.T)
        #print "output",
        #print self.output,
        #print sign(self.output)

        return self.output

    def BackPropagation(self):
        teacher = self.input
        #oldw_h2o = self.w_h2o
        
        delta_h2o = (teacher-self.output) * self.output * (1.-self.output)
        e1 = np.array(np.mat(delta_h2o).T * np.mat(self.hidden))
        self.w_i2o = self.w_i2o + self.alpha * e1.T

        #e2 = delta_h2o
        #self.b_o = self.b_o + self.alpha * e2
        
        #delta_i2h = self.hidden * (1.-self.hidden) * np.dot(self.w_h2o.T , delta_h2o)
        delta_i2h = self.hidden * (1.-self.hidden) * np.dot(self.w_i2o , delta_h2o)
        
        f1 = np.array(np.mat(delta_i2h).T * np.mat(self.input))
        self.w_i2o = self.w_i2o + self.alpha * f1
        
        #f2 = delta_i2h
        #self.b_h = (self.b_h + self.alpha * f2)

    def Train(self,data):
        self.ForwardPropagation(data)
        self.BackPropagation()
        #self.printw()
        
    def config(self):
        print u"number input:%d hidden:%d output:%d" % (self.n_input,self.n_output)
        print u"i2o:%d" % (self.n_input*self.n_output)
        print self.w_i2o
        print self.b_o
    def printw(self):
        print self.w_i2o

def train(loop=10000):
    nn = AutoEncoder(n_input=4, n_output=3)
    data1 = np.array([0,0,0,0])
    data2 = np.array([1,0,0,1])
    data3 = np.array([0,0,1,0])
    data4 = np.array([1,0,1,1])
    data5 = np.array([0,1,0,0])
    data6 = np.array([1,1,0,1])
    data7 = np.array([0,1,1,0])
    data8 = np.array([1,1,1,1])
    data9 = np.array([1,0,0,0])
    data10 = np.array([0,0,0,1])
    data11 = np.array([1,0,1,0])
    data12 = np.array([0,0,1,1])
    data13 = np.array([1,1,0,0])
    data14 = np.array([0,1,0,1])
    data15 = np.array([1,1,1,0])
    data16 = np.array([0,1,1,1])
    for i in xrange(loop):
        nn.Train(data1)
        nn.Train(data2)
        nn.Train(data3)
        nn.Train(data4)
        nn.Train(data5)
        nn.Train(data6)
        nn.Train(data7)
        nn.Train(data8)
        nn.Train(data9)
        nn.Train(data10)
        nn.Train(data11)
        nn.Train(data12)
        nn.Train(data13)
        nn.Train(data14)
        nn.Train(data15)
        nn.Train(data16)
    return nn

def test(nn):
    data1 = np.array([0,0,0,0])
    data2 = np.array([0,0,0,1])
    data3 = np.array([0,0,1,0])
    data4 = np.array([0,0,1,1])
    data5 = np.array([0,1,0,0])
    data6 = np.array([0,1,0,1])
    data7 = np.array([0,1,1,0])
    data8 = np.array([0,1,1,1])
    data9 = np.array([1,0,0,0])
    data10 = np.array([1,0,0,1])
    data11 = np.array([1,0,1,0])
    data12 = np.array([1,0,1,1])
    data13 = np.array([1,1,0,0])
    data14 = np.array([1,1,0,1])
    data15 = np.array([1,1,1,0])
    data16 = np.array([1,1,1,1])
    print nn.ForwardPropagation(data1)
    print sign(nn.ForwardPropagation(data1))
    print nn.ForwardPropagation(data2)
    print sign(nn.ForwardPropagation(data2))
    print nn.ForwardPropagation(data3)
    print sign(nn.ForwardPropagation(data3))
    print nn.ForwardPropagation(data4)
    print sign(nn.ForwardPropagation(data4))
    print nn.ForwardPropagation(data5)
    print sign(nn.ForwardPropagation(data5))
    print nn.ForwardPropagation(data6)
    print sign(nn.ForwardPropagation(data6))
    print nn.ForwardPropagation(data7)
    print sign(nn.ForwardPropagation(data7))
    print nn.ForwardPropagation(data8)
    print sign(nn.ForwardPropagation(data8))
    print nn.ForwardPropagation(data9)
    print sign(nn.ForwardPropagation(data9))
    print nn.ForwardPropagation(data10)
    print sign(nn.ForwardPropagation(data10))
    print nn.ForwardPropagation(data11)
    print sign(nn.ForwardPropagation(data11))
    print nn.ForwardPropagation(data12)
    print sign(nn.ForwardPropagation(data12))
    print nn.ForwardPropagation(data13)
    print sign(nn.ForwardPropagation(data13))
    print nn.ForwardPropagation(data14)
    print sign(nn.ForwardPropagation(data14))
    print nn.ForwardPropagation(data15)
    print sign(nn.ForwardPropagation(data15))
    print nn.ForwardPropagation(data16)
    print sign(nn.ForwardPropagation(data16))
                
def train2(loop=1000):
    nn = AutoEncoder(n_input=8, n_output=6)
    data1 = np.array([0,0,0,0,1,1,1,1])
    data2 = np.array([0,1,0,0,1,1,0,0])
    data3 = np.array([0,0,0,1,1,0,0,1])
    data4 = np.array([1,0,1,0,1,0,0,0])
    data5 = np.array([1,1,0,1,1,0,0,0])
    data6 = np.array([0,1,0,1,0,1,0,1])
    data7 = np.array([0,1,1,0,0,1,0,0])
    data8 = np.array([0,0,0,0,0,0,0,0])
    data9 = np.array([0,0,0,0,1,0,0,0])
    data10 = np.array([1,0,0,0,0,0,0,0])
    data11 = np.array([1,0,1,0,0,1,1,0])
    data12 = np.array([0,1,1,1,0,0,0,1])
    data13 = np.array([1,1,0,0,0,0,1,0])
    data14 = np.array([0,0,1,1,1,1,0,0])
    data15 = np.array([0,0,1,0,1,0,0,0])
    data16 = np.array([0,1,0,1,1,1,1,1])
    for i in xrange(loop):
        nn.Train(data1)
        nn.Train(data2)
        nn.Train(data3)
        nn.Train(data4)
        nn.Train(data5)
        nn.Train(data6)
        nn.Train(data7)
        nn.Train(data8)
        nn.Train(data9)
        nn.Train(data10)
        nn.Train(data11)
        nn.Train(data12)
        nn.Train(data13)
        nn.Train(data14)
        nn.Train(data15)
        nn.Train(data16)
    return nn

def test2(nn):
    data1 = np.array([0,0,0,0,1,1,1,1])
    data2 = np.array([0,1,0,0,1,1,0,0])
    data3 = np.array([0,0,0,1,1,0,0,1])
    data4 = np.array([1,0,1,0,1,0,0,0])
    data5 = np.array([1,1,0,1,1,0,0,0])
    data6 = np.array([0,1,0,1,0,1,0,1])
    data7 = np.array([0,1,1,0,0,1,0,0])
    data8 = np.array([0,0,0,0,0,0,0,0])
    data9 = np.array([0,0,0,0,1,0,0,0])
    data10 = np.array([1,0,0,0,0,0,0,0])
    data11 = np.array([1,0,1,0,0,1,1,0])
    data12 = np.array([0,1,1,1,0,0,0,1])
    data13 = np.array([1,1,0,0,0,0,1,0])
    data14 = np.array([0,0,1,1,1,1,0,0])
    data15 = np.array([0,0,1,0,1,0,0,0])
    data16 = np.array([0,1,0,1,1,1,1,1])
    print nn.ForwardPropagation(data1)
    print sign(nn.ForwardPropagation(data1))
    print nn.ForwardPropagation(data2)
    print sign(nn.ForwardPropagation(data2))
    print nn.ForwardPropagation(data3)
    print sign(nn.ForwardPropagation(data3))
    print nn.ForwardPropagation(data4)
    print sign(nn.ForwardPropagation(data4))
    print nn.ForwardPropagation(data5)
    print sign(nn.ForwardPropagation(data5))
    print nn.ForwardPropagation(data6)
    print sign(nn.ForwardPropagation(data6))
    print nn.ForwardPropagation(data7)
    print sign(nn.ForwardPropagation(data7))
    print nn.ForwardPropagation(data8)
    print sign(nn.ForwardPropagation(data8))
    print nn.ForwardPropagation(data9)
    print sign(nn.ForwardPropagation(data9))
    print nn.ForwardPropagation(data10)
    print sign(nn.ForwardPropagation(data10))
    print nn.ForwardPropagation(data11)
    print sign(nn.ForwardPropagation(data11))
    print nn.ForwardPropagation(data12)
    print sign(nn.ForwardPropagation(data12))
    print nn.ForwardPropagation(data13)
    print sign(nn.ForwardPropagation(data13))
    print nn.ForwardPropagation(data14)
    print sign(nn.ForwardPropagation(data14))
    print nn.ForwardPropagation(data15)
    print sign(nn.ForwardPropagation(data15))
    print nn.ForwardPropagation(data16)
    print sign(nn.ForwardPropagation(data16))
    
if __name__ == "__main__":
    nn = train()
    test(nn)

        
