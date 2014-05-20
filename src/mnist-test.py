# coding: utf-8
import os
import gzip
import pylab
#import cPickle
import pickle
import numpy as np
from network import Network
from sklearn.preprocessing import LabelBinarizer

def load_mnist():
    with gzip.open('../data/mnist.pkl.gz', 'rb')  as f:
        train_set, valid_set, test_set = pickle.load(f)
    return train_set, valid_set, test_set

if __name__ == "__main__":
    # initialize
    train_set, valid_set, test_set = load_mnist()
    train_data, train_target = train_set
    valid_data, valid_target = valid_set
    test_data, test_target = test_set
    train_target = LabelBinarizer().fit_transform(train_target)
    valid_target = LabelBinarizer().fit_transform(valid_target)
    test_target = LabelBinarizer().fit_transform(test_target)
    
    # size
    train_size = 50000
    valid_size = 10000
    test_size = 10000

    # train
    epoch = 100
    nn = Network([784, 1500, 700, 10])
    for e in xrange(epoch):
        print "epoch:%d" % e
        for i in xrange(train_size):
            nn.train(train_data[i], train_target[i])
        
        #"""
        correct = 0
        for i in xrange(test_size):
            output = nn.forward_propagation(test_data[i])
            if np.argmax(output) == np.argmax(test_target[i]):
                correct += 1
        print u"correct: %d / %d" % (correct, test_size)
        #"""

    # test
    correct = 0
    for i in xrange(test_size):
        output = nn.forward_propagation(test_data[i])
        if np.argmax(output) == np.argmax(test_target[i]):
            correct += 1
        else:
            print u"error: %d & %d" % (np.argmax(output), 
                                       np.argmax(test_target[i]))
    print u"correct: %d / %d" % (correct, test_size)

    
            
