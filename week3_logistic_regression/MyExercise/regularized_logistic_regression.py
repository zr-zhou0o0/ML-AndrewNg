import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def get_data():
    '''
    X1,X2,Y_target: shape=(m,)=(118,), dtype=float64, type=series
    '''
    df = pd.read_csv('ex2data2.txt', header=None, names=['test1', 'test2', 'ac'])
    print(df.describe())
    X1 = df['test1']
    X2 = df['test2']
    Y_target = df['ac']
    return X1, X2, Y_target, df


def feature_x(X1, X2):
    '''
    X1,X2: shape=(m,)=(118,), type=float64
    X: shape=(m,p)=(118,28), type=float64
    X = [1, X1, X2, X1^2, X1*X2, X2^2, X1^3, ..., X1*X2^5, X2^6]
    
    # arrays can compute element-wise exponentiation, while 
    # matrices can only perform matrix operations for exponentiation.
    # print(f"X1={X1[0:5]}, test={(X1**2)[0:5]}")
    '''
    m = len(X1)
    ones = np.ones((m,1)) # not ones_like
    X = ones
    power = 6
    for i in range(1, power+1, 1):
        # print(f"################i={i}################")
        for j in range(i, -1, -1):
            # print(f"-----------j={j}------------")
            X1_temp = X1 ** j
            X2_temp = X2 ** (i-j)
            X_temp = np.multiply(X1_temp, X2_temp)
            X_temp = np.matrix(X_temp).reshape(m,-1)
            # print(f"Xtemp={X_temp[0:5]}")
            X = np.concatenate((X,X_temp), axis=-1)
            # print(f"X={X[0:5]}")
    # print(f"Xshape={X.shape}") # (118,28)
    return X


def plot(df, theta, mode):
    positive = df[df['ac'].isin([1])]
    negative = df[df['ac'].isin([0])]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(positive['test1'], positive['test2'], s=10, c='c', marker='o', label='Accepted')
    ax.scatter(negative['test1'], negative['test2'], s=10, c='b', marker='o', label='rejected')
    ax.legend()
    ax.set_xlabel('test1')
    ax.set_ylabel('test2')
    if mode == 0:
        # plt.show()
        plt.savefig('dataset2')
    if mode == 1:
        X1 = np.arange(-1, 1, 0.01)
        X2 = np.arange(-1, 1, 0.01)
        m_sample = len(X1)
        X1, X2 = np.meshgrid(X1, X2)

        # X = feature_x(X1, X2)
        # print(f"X1_0={X1[:,0]}")
        # print(f"X1_0shape={X1[:,0].shape}")  # shape=(200,)
        # print(f"Xshape={X.shape}")

        Z = np.ones((m_sample,1)) # ones_like(m,1) output (1,1)
        for i in range(1, m_sample, 1):
            Xi = feature_x(X1[:,i], X2[:,i])  # shape=(200,28)
            Zi = np.dot(Xi, theta)  # shape=(1,200)
            Zi = Zi.reshape(m_sample,-1)  # shape=(200,1)
            # print(f"Zishape={Zi.shape}")
            Z = np.concatenate((Z,Zi), axis=-1)
            # print(f"Zshape={Z.shape}")

        plt.contour(X1, X2, Z, 0)
        # plt.show()
        plt.savefig('logisitic_regression_regularization')     


def sigmoid(theta, X):
    '''
    p = 28 #parameters
    m = 118
    theta: shape=(p,)
    theta_T: shape=(p,1)
    X: shape=(m,p)
    Z: shape=(m,1)
    Y: shape=(m,1)
    '''
    p = len(theta)
    theta_t = theta.reshape(p,1)
    Z = np.dot(X, theta_t)  # forgot to write return, resulting in X being NoneType
    Y = 1/(1 + np.exp(-Z))
    return Y


def cost(theta, X, Y_target, rgl_lambda, learningRate):
    '''
    p = 28
    m = 118
    theta: shape=(p,)
    X: shape=(m,p)
    Y_target: shape=(m,)
    Y: shape=(m,1)
    '''
    m = X.shape[0]
    Y = sigmoid(theta, X)

    # To convert a matrix to 0-dimensional, first need to convert it to an array.
    # Otherwise it will always remain in (1, 118) or (118, 1) shape.
    # print(Y.shape)  # (118,1)
    # print(np.array(Y).shape)  # (118,1)
    # Y = np.array(Y)  
    # print(Y.flatten().shape)  # matrix:(1,118) array:(118,)
    # print(np.squeeze(Y).shape)  # matrix:(1,118) array:(118,)

    Y = np.array(Y)
    cost1 = -np.multiply(Y_target, np.log(Y).reshape(-1)) # shape=(m,)
    cost2 = -np.multiply(1-Y_target, np.log(Y).reshape(-1))
    costs = (np.sum(cost1 + cost2))/m
    theta0 = theta
    theta0[0] = 0
    punish = theta0 ** 2
    punishs = np.sum(punish) * rgl_lambda/(2*m)
    cost = costs + punishs
    # print(cost)
    return cost


def gradient(theta, X, Y_target, rgl_lambda, learningRate):
    '''
    p = 28
    m = 118
    theta: shape=(p,)
    X: shape=(m,p)
    Y_target: shape=(m,); shape=(1,m)
    Y: shape=(m,1); shape=(1,m)
    '''
    m = X.shape[0]
    Y = sigmoid(theta, X)
    Y = Y.flatten()
    Y_target = np.matrix(Y_target)
    # print(f"Yshape={Y.shape}, Y_targetshape={Y_target.shape}")
    delta_Y = Y - Y_target
    grad1 = np.dot(delta_Y, X)/m
    theta0 = theta
    theta0[0] = 0
    # print(theta0)
    grad2 = theta0 * rgl_lambda/m
    grad = grad1 + grad2
    return grad


def main():
    X1, X2, Y_target, df = get_data()
    
    X = feature_x(X1, X2)
    theta = np.zeros(28)
    rgl_lambda = 1
    learningRate = 0.1

    plot(df, theta, mode=0)

    # Y = sigmoid(theta, X)
    # print(f"Y={Y}")
    # cost(theta, X, Y_target, rgl_lambda, learningRate)
    # gradient(theta, X, Y_target, rgl_lambda, learningRate)

    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, Y_target, rgl_lambda, learningRate))
    print(result)

    theta_min = result[0]
    plot(df, theta_min, mode=1)

    # print(Y_target.shape)  # series, shape=(118,)
    # print(np.array(Y_target).shape)  # array, shape=(118,)


if __name__ == '__main__':
    main()