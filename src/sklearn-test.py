# coding: utf-8
import numpy as np
from network import Network
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer

if __name__ == "__main__":
    # initialize
    digits = load_digits()

    data = digits.data
    data /= data.max()

    target = digits.target
    target = LabelBinarizer().fit_transform(target)
    
    # train & test size
    train_size = 1197
    test_size = 1797 - train_size

    # train
    epoch = 100
    nn = Network([64, 100, 10])
    for e in xrange(epoch):
        print u"epoch:%d" % e
        for i in xrange(train_size):
            nn.train(data[i], target[i])
        #"""
        correct = 0
        for i in xrange(test_size):
            output = nn.forward_propagation(data[i + train_size])
            if np.argmax(output) == np.argmax(target[i + train_size]):
                correct += 1
        print u"correct: %d / %d" % (correct, test_size)
        #"""

    # test
    correct = 0
    for i in xrange(test_size):
        output = nn.forward_propagation(data[i + train_size])
        #print u"o:%d t:%d" % (np.argmax(output), np.argmax(target[i]))
        if np.argmax(output) == np.argmax(target[i + train_size]):
            correct += 1
        else:
            print u"error: %d & %d" % (np.argmax(output), np.argmax(target[i + train_size]))
    print u"correct: %d / %d" % (correct, test_size)
