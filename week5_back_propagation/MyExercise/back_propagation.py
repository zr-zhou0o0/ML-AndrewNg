# INCOMPLETE
# NOT RUN YET

'''
- attributes:
    m = number of samples; m=5000
    p = number of parameters, with biases; p1=401, p2=26
    K = number of categories/units; K1=25, K2=10
    theta1 = parameters1, theta2 = parameters2, with ones
- params:
    layer0(inputlayer): -> (5000,401) with 1 in the first colomn
    layer1(hiddenlayer): (5000,401) -> (25,401) -> (5000,26) with 1 in the first colomn
    layer2(outputlayer): (5000,26) -> (10,26) -> (5000,10)

'''

import numpy as np
import matplotlib.pylab as plt
import scipy
from scipy.io import loadmat
from scipy.optimize import minimize
import sys

np.printoptions(threshold=sys.maxsize)

test_idx = 2345

def get_data(mode=0, idx=test_idx):
    '''
    mode=0: no image
    mode=1: the idx image 
    X: shape=(5000,400)
    y: shape=(5000,1)
    '''
    data = loadmat('ex4data1')
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


def get_theta():
    '''
    theta1: shape=(25,401)
    theta2: shape=(10,26)
    '''
    theta = scipy.io.loadmat('ex4weights.mat')
    # print(theta)
    print(theta['Theta1'].shape)
    print(theta['Theta2'].shape)
    theta1 = theta['Theta1']
    theta2 = theta['Theta2']
    return theta1, theta2


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


def sigmoid(theta, A):
    '''
    theta: shape=(K1,p1) b, w_1, w_2, ...
    theta_T: shape=(p1,K1)
    A: shape=(m,p1) with the first colomn is 'ones'
    Y: shape=(m,K1)
    return Y
    '''

    theta_T = theta.T
    Z = np.dot(A, theta_T)
    Y = 1/(1+np.exp(-Z))
    return Y


def sigmoid_gradient(theta, A):
    return np.multiply(sigmoid(theta, A), (1 - sigmoid(theta, A)))


def forward_propagate(params, X):
    '''
    params: an array contains flattened theta1 concatenates with flattened theta2 (no)
    params: a list contains parameter matrices (yes)
    X: shape=(m,p) with 1 at first colomn
    '''
    theta1 = params[0]
    theta2 = params[1]
    theta3 = params[2]

    a1, a2, a3 = sequential(X, theta1, theta2, theta3)
    return a1, a2, a3



def cost(params, X, Y_target, learning_rate):

    m = X.shape[0]
    X = np.matrix(X)
    Y_target = np.matrix(Y_target)
    theta1 = params[0]
    theta2 = params[1]
    
    _,category,_ = forward_propagate(params, X)
    
    J = 0
    for i in range(m):
        first_term = np.multiply(-Y_target[i,:], np.log(category[i,:]))
        second_term = np.multiply((1 - Y_target[i,:]), np.log(1 - category[i,:]))
        J += np.sum(first_term - second_term)
    J = J / m
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    
    return J




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
    a_in: shape=(m,p-1) -> (m,p)(deprecate)
    a_in: shape=(m,p)
    theta: shape=(K,p) 
    a_out: shape=(m,K+1)
    func: activation function
    '''
    m = a_in.shape[0]
    p = a_in.shape[1]
    theta = np.matrix(theta).reshape(-1,p)
    K = theta.shape[0]

    # a_in = np.insert(a_in, 0, values=np.ones(m), axis=1)

    a_out = func(theta, a_in)
    a_out = np.insert(a_out, obj=0, values=np.ones(m), axis=1)

    return a_out


def sequential(X, theta1, theta2, theta3):
    a1 = dense(X, theta1, sigmoid)
    a2 = dense(a1, theta2, sigmoid)
    a3 = dense(a2, theta3, prob_max)
    return a1, a2, a3



def backprop(params, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    theta1 = params[0]
    theta2 = params[1]

    # run the feed-forward pass
    # a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # a0=(5000,401), a1=(5000,26), a2=(5000,11), a3=(5000,2)
    a0 = np.insert(X, 0, values=np.ones(m), axis=1)
    a1, a2, a3 = forward_propagate(params, X)
    z1 = a1[:, 1:] # wrong. sigmoid(z1) = a1[:, 1:]
    z2 = a2[:, 1:] # wrong
    z3 = a3[:, 1:] # wrong
    
    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    J = cost(params, X, y, learning_rate)
   
    # perform backpropagation
    for t in range(m):
        a0t = a0[t,:]  # (1, 401)
        z1t = z1[t,:]  # (1, 25)
        a1t = a1[t,:]  # (1, 26)
        z2t = z2[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)
        
        d2t = z2t - yt  # (1, 10)
        
        # z1t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d1t = np.multiply((d2t, theta2), sigmoid_gradient(a1t))  # (1, 26) 
        
        delta1 = delta1 + (d1t[:,1:]).T * a0t # (25, 401)
        delta2 = delta2 + d2t.T * a1t # (10, 26)
        
    delta1 = delta1 / m
    delta2 = delta2 / m
    
    # add the gradient regularization term
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m
    
    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    
    return J, grad

def gradient_descent():
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


def main():

    X, Y_target = get_data()
    Y_encoded = feature_y(Y_target, 10)

    theta1, theta2 = get_theta()
    theta3 = np.zeros((1,11))
    # theta1, theta2 should have the ones at first colomn

    params = []
    params[0] = theta1
    params[1] = theta2
    params[2] = theta3

    learning_rate = 0.1

    fmin = minimize(fun=backprop, x0=params, args=(params, X, Y_encoded, learning_rate), 
                method='TNC', jac=True, options={'maxiter': 250})

    X = np.matrix(X)
    theta1 = fmin.x[0]
    theta2 = fmin.x[1]

    _,_,y_pred = forward_propagate(X, theta1, theta2)
    
    # category = sequential(X, theta1, theta2, theta3)
    accuracy = compute_precision(y_pred, Y_target)
    print(f"accuracy={accuracy}")



if __name__ == '__main__':
    main()
