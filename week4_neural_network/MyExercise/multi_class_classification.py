'''
m = samples
p = parameters
K = classes
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
import sys

test_idx = 2345

def get_data(mode=0, idx=test_idx):
    '''
    mode=0: no image
    mode=1: the idx image 
    X: shape=(5000,400)
    y: shape=(5000,1)
    '''
    data = loadmat('ex3data1')
    X = data['X']
    y = data['y']
    # print(data)
    print(data['X'].shape, data['y'].shape)
    # one image is 20*20=400 pixels, y is the true digits
    if mode==1:
        image = data['X'][idx].reshape(20,20)
        # print(f"image shape={image}")
        plt.imshow(image, cmap='gray')
        # plt.show(block=False)
        plt.ion()
        plt.savefig('test')
        plt.pause(3)
        plt.close()
    return X, y


def feature_x(X):
    '''
    p = 401
    m = 5000
    X: shape=(m,p-1) -> shape=(m,p)
    '''
    m = X.shape[0]
    p = X.shape[1]+1
    ones = np.ones((m,1))
    X = np.concatenate((ones, X), axis=-1)
    return X


def feature_y(Y_target, K):
    '''
    one-hot encoder
    Y_target: shape=(m,1)
    Y: shape=(m,K) with label K = 1, 2, ..., 9, 10(0)
    note: category 1, index 0
    '''
    m = Y_target.shape[0]
    Y = np.zeros(m)
    Y = np.matrix(Y).reshape(m,1)
    for i in range(1, K+1):
        # if else and for in combined.
        # 'label' traverse the elements in Y_target.
        Y_i = np.array([1 if label == i else 0 for label in Y_target])
        Y_i = np.matrix(Y_i).reshape(m,1)
        Y = np.concatenate((Y, Y_i), axis=-1)
    Y = np.delete(Y, 0, axis=-1)
    # print(f"yshape={Y.shape}")
    # print(f"yoriginal test={Y_target[test_idx]}")
    # print(f"y test={Y[test_idx]}")
    return Y


def sigmoid(theta, X):
    '''
    theta: shape=(K,p) b, w_1, w_2, ...
    theta_T: shape=(p,K)
    X: shape=(m,p) with the first colomn is 'ones'
    Y: shape=(m,K)
    return Y
    '''
    # print(f"theta in sigmoid shape:{theta.shape},len{len(theta.shape)}")
    
    # if len(theta.shape) == 2:
    #     p = theta.shape[1]  
    #     K = theta.shape[0]
    # elif len(theta.shape) == 1:
    #     P = len(theta)
    #     K = 1
    # else:
    #     raise ValueError("Invalid shape for theta.")

    theta_T = theta.T
    Z = np.dot(X, theta_T)
    Y = 1/(1+np.exp(-Z))
    return Y


def cost(theta, X, Y_target, rgl_lambda):
    '''
    theta: shape=(K,p)
    X: shape=(m,p)
    Y: shape=(m,K)
    Y_target(encoded): shape=(m,K)
    costs: shape=(1,K)
    '''
    m = X.shape[0]
    p = X.shape[1]

    theta = np.matrix(theta).reshape(-1,p)
    Y_target = np.matrix(Y_target).reshape(m,-1)
    K = Y_target.shape[1]

    # if len(Y_target.shape) == 2:
    #     K = Y_target.shape[1]
    # elif len(Y_target.shape) == 1:
    #     K = 1
    # else:
    #     raise ValueError("Invalid shape for theta.")
    
    Y = sigmoid(theta, X)

    # print(f"Ytargetshape={Y_target.shape}")

    assert Y_target.shape == (5000,1)
    assert Y.shape == (5000,1)

    cost1 = -np.multiply(Y_target, np.log(Y))
    ### here!!! erroneously take log(1-Y) as log(Y)!!!
    cost2 = -np.multiply(1-Y_target, np.log(1-Y)) 
    costs = (np.sum((cost1 + cost2), axis=0))/m

    # print(f"costs={costs}")

    theta0 = theta
    theta0[:,0] = 0
    punish = np.square(theta0)
    punishs = (np.sum(punish, axis=-1) * rgl_lambda/(2*m)).T
    cost = costs + punishs
    assert cost.shape == (1,K)
    # print(f"costshape={cost.shape}")
    # print(f"cost={cost}")
    return cost


def gradient(theta, X, Y_target, rgl_lambda):
    '''
    theta: shape=(K,p) **
    X: shape=(m,p) **
    Y: shape=(m,K)
    Y_target: shape=(m,K)
    delta_Y: shape=(m,K)
    grad: shape=(K,p)
    '''
    m = X.shape[0]
    p = X.shape[1]
    theta = np.matrix(theta).reshape(-1,p)
    Y_target = np.matrix(Y_target).reshape(m,-1)
    K = theta.shape[0]

    Y = sigmoid(theta, X)
    delta_Y = Y - Y_target
    # print(f"Yshape={Y.shape}, Y_targetshape={Y_target.shape}")

    assert theta.shape == (K,p)
    assert delta_Y.shape == (m,K)
    grad1 = np.dot(delta_Y.T, X)/m

    theta0 = theta
    theta0[:,0] = 0
    grad2 = theta0 * rgl_lambda/m
    grad = grad1 + grad2
    assert grad.shape == (K,p)
    # print(f"gradshape={grad.shape}")
    # print(f"grad truncated:{grad[0,0:10]}")
    return grad


# not the Gradient Function Issue!!!
# however the gradient calculates wrong.
# it is an external issue.
# y is same, X is same...
# change the one_vs_all, the gradients become correct however costs are still wrong!
# which means Cost Function Issue AND External Issue
# cost function does indeed have an issue. wrote a wrong formula. 
# select wrong index in function one_vs_all...


def one_vs_all(theta, X, Y_target, rgl_lambda, learningRate):
    '''
    theta: shape=(K,p)
    X: shape=(m,p) 
    Y: shape=(m,K)
    Y_target: shape=(m,K)
    cost: shape=(1,K)
    grad: shape=(K,p)
    '''
    np.set_printoptions(threshold=sys.maxsize)
    K = theta.shape[0]
    m = X.shape[0]
    p = X.shape[1]
    theta_min = np.zeros((K,p))

    # i is index, i=0,1,2,...,9
    for i in range(K):
        fmin = minimize(fun=cost, x0=theta[i,:], args=(X, Y_target[:,i], rgl_lambda), method='TNC', jac=gradient)
        theta_min[i,:] = fmin.x # 1234567890
        # print(f"i={i}, fminx={fmin.x}")
    
    # explicit definition: i is the index or the value of the number???
    # i=3
    # theta_i = theta[2,:] # i=3 -> index=2 !!!!!
    # Y_target_i = Y_target[:,2]

    # fmin = minimize(fun=cost, x0=theta_i, args=(X, Y_target_i, rgl_lambda), method='TNC', jac=gradient)
    # print(f"fminx={fmin.x}")
    # theta_min[i-1,:] = fmin.x

    return theta_min


def predict_all(theta_min, X):
    '''
    theta_min: shape=(K,p)
    X: shape=(m,p)
    prob: shape=(m,K), suggest the probability
    note: index 0~9, number 1~10
    '''
    m = X.shape[0]
    prob = sigmoid(theta_min, X)
    maxprob = np.amax(prob, axis=1).reshape(m,-1)
    category = np.argmax(prob, axis=1).reshape(m,-1) +1
    # print(category)
    return category
    pass


def compute_precision(category, Y_target):
    '''
    category: shape=(m,1), 1~10
    Y_target: not encoded, shape=(m,1), 1~10
    '''
    # print(Y_target)
    m = category.shape[0]
    correct = np.array([1 if category[i] == Y_target[i] else 0 for i in range(m)])
    accuracy = np.sum(correct)/m
    return accuracy

    pass


def main():

    # test function
    K = 10

    X, Y_target = get_data(mode=0)
    X = feature_x(X)
    Y_target_encoded = feature_y(Y_target, K)

    p = X.shape[1]
    m = X.shape[0]
    learningRate = 0.1
    rgl_lambda = 1

    theta = np.zeros((K,p))
    # print(f"theta shape={theta.shape}")

    # cost(theta, X, Y_target_encoded, rgl_lambda, learningRate)
    # grad = gradient(theta, X, Y_target_encoded, rgl_lambda, learningRate)
    theta_min = one_vs_all(theta, X, Y_target_encoded, rgl_lambda, learningRate)
    print(theta_min)

    category = predict_all(theta_min, X)
    accuracy = compute_precision(category, Y_target)
    print(f"accuracy={accuracy}")

    pass


if __name__ == '__main__':
    main()