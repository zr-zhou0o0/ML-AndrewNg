'''
- attributes:
    m = samples
    p = parameters, with biases
    K = categories/units
'''

import numpy as np
import matplotlib.pylab as plt
import scipy
import sys

import multi_class_classification as mcc

np.printoptions(threshold=sys.maxsize)

def get_theta():
    '''
    theta1: shape=(25,401)
    theta2: shape=(10,26)
    '''
    theta = scipy.io.loadmat('ex3weights.mat')
    # print(theta)
    print(theta['Theta1'].shape)
    print(theta['Theta2'].shape)
    theta1 = theta['Theta1']
    theta2 = theta['Theta2']
    return theta1, theta2


def prob_max(theta, X):
    '''
    X: shape=(m,p)
    return: category, shape=(m,1), 1~10
    '''
    X = np.delete(X, 0, axis=1)
    m = X.shape[0]
    category = np.argmax(X, axis=1).reshape(m,-1) +1
    return category


def dense(a_in, theta, func):
    '''
    a_in: shape=(m,p-1) -> (m,p)
    theta: shape=(K,p) 
    a_out: shape=(m,K)
    func: activation function
    '''
    m = a_in.shape[0]
    p = a_in.shape[1]+1
    theta = np.matrix(theta).reshape(-1,p)
    K = theta.shape[0]

    a_in = np.insert(a_in, 0, values=np.ones(m), axis=1)

    a_out = func(theta, a_in)
    return a_out


def sequential(X, theta1, theta2, theta3):
    a1 = dense(X, theta1, mcc.sigmoid)
    a2 = dense(a1, theta2, mcc.sigmoid)
    a3 = dense(a2, theta3, prob_max)
    return a3


def main():

    X, Y_target = mcc.get_data()
    theta1, theta2 = get_theta()
    theta3 = np.zeros((1,11))

    category = sequential(X, theta1, theta2, theta3)
    accuracy = mcc.compute_precision(category, Y_target)
    print(f"accuracy={accuracy}")

    pass


if __name__ == '__main__':
    main()
