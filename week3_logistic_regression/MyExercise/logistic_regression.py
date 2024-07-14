import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def get_data():
    '''
    X shape = (100, 2)
    y shape = (100, 1)
    '''
    dataframe = pd.read_csv('ex2data1.txt', header=None, names=['Exam1', 'Exam2', 'Admitted'])
    # print(dataframe.describe())
    # x1 = dataframe['Exam1']
    # x2 = dataframe['Exam2']
    X = dataframe.iloc[:, 0:2]
    y = dataframe['Admitted']
    m = X.shape[0]

    # feature scaling in case of data overflow
    # it works!!!
    X /= 100
    dataframe = pd.merge(X, y, left_index=True, right_index=True)
    # print(dataframe)

    X = np.matrix(X.values)
    y = np.matrix(y.values).reshape(100,1)  # or it will be (1, 100)
    # print("1:", y.shape)
    # print(m)
    return X, y, m, dataframe
    

def plot_data(df, W, b, mode):
    '''
    mode=0: without boundary line
    mode=1: with boundary line
    w1*x1 + w2*x2 + b = 0
    '''
    fig, ax = plt.subplots(figsize=(6,4))
    positive = df[df['Admitted'].isin([1])]
    negative = df[df['Admitted'].isin([0])]
    ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='m', marker='o', label='Admitted')
    ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='c', marker='x', label='Not Admitted')
    ax.legend()  # 图例
    ax.set_xlabel('Exam 1 score')
    ax.set_ylabel('Exam 2 score')
    if mode == 0:
        # plt.show()
        plt.savefig("dataset")
    if mode == 1:
        print(f"plot W={W} b={b}")
        # print(W[0,0])
        x1 = np.arange(0,1,0.1)
        B = b * np.ones(len(x1))
        x2 = (-B - W[0,0] * x1)/W[0,1]  # it is W[0,0], not W[0][0] !!!
        ax.plot(x1, x2, c='b')
        # plt.show()
        plt.savefig("logistic_regression")

    # print(positive)


def model(X, Y_target, m, W, b):
    '''
    Y_predict shape = (m,1)

    - np.multiply 元素乘法
    - np.dot 矩阵乘法
    - * array是元素乘法 matrix是矩阵乘法
    '''
    W = W.T
    Z = np.dot(X, W)  # shape = (1, 100)
    Z = Z.reshape(m, 1)
    # print(Z.shape)
    # print(Z)
    B = b * np.ones(m)  # shape = (m,)
    B = B.reshape(m,1)  # shape = (m, 1)
    Z = Z + B
    # print(f"model_z={Z}")
    Y_predict = 1.0 / (1.0 + np.exp(-Z))  # shape = (m, 1)
    # print(Y.shape)
    return Y_predict


def cost_function(X, Y_target, Y_predict, m, W, b):
    cost = 0.0
    X = np.matrix(X)
    Y_target = np.matrix(Y_target)
    Y_predict = np.matrix(Y_predict)
    cost1 = - np.multiply(-Y_target, np.log(Y_predict))
    cost1s = np.sum(cost1)
    cost2 = - np.multiply(1-Y_target, np.log(1-Y_predict))
    cost2s = np.sum(cost2)
    cost = (cost1s + cost2s)/m

    # for i in range(m):
    #     # 对于matrix，*代表矩阵乘法而不是元素相乘
    #     # 但predict等于1应该是因为数据溢出了
    #     loss = -Y_target[i] * np.log(Y_predict[i]) - (1-Y_target[i]) * np.log(1-Y_predict[i])
    #     cost += loss
    # cost /= m

    return cost
    

# 如果要用scipy的话
# def cost(theta, X, y):


def compute_gradient_descent(X, Y_target, Y_predict, m, W, b, lr):
    # theta = np.concatenate((W,b), axis=0)
    # print(f"win={W},bin={b}")
    theta = np.append(W, b)
    # print(f"theta={theta}")
    ones = np.ones(m)
    ones = ones.reshape(m,1)
    X1 = np.concatenate((X, ones), axis=1)
    # print(X1)

    dj_dt = np.dot((Y_predict-Y_target).T, X1)/m
    # return dj_dt
    theta_new = theta - lr * dj_dt
    # print(f"ttn={theta_new}")
    W_new = theta_new[0, 0:2] # theta shape(1,3) not(3,)
    b_new = theta_new[0, 2]
    W_new = np.array(W_new)
    # print(f"Wn={W_new}, bn={b_new}")

    # err_sum = np.dot((Y_predict - Y_target).reshape(1, m), X)
    # dj_dw = err_sum/m  # shape = (1,2)
    # W_new = W-lr*dj_dw
    # # print(W_new)
    # dj_db = (np.sum(Y_predict - Y_target))/m
    # b_new = b - lr*dj_db  # scalar
    return W_new, b_new # W shape (2, )


def run_gradient_descent(X, Y_target, m, W_in, b_in, lr, iters):
    J_history = []
    W = copy.deepcopy(W_in)
    b = b_in
    for i in range(iters):
        Y_predict = model(X, Y_target, m, W, b)
        cost = cost_function(X, Y_target, Y_predict, m, W, b)
        J_history.append(cost)
        W_temp, b_temp = compute_gradient_descent(X, Y_target, Y_predict, m, W, b, lr)
        W = W_temp
        b = b_temp
        if (i%(math.ceil(iters/10))) == 0:
            # print(f"Y_predict={Y_predict[1:10]}")
            print(f"iter={i} cost={cost} W={W} b={b}")
            # print("iter={} cost={}".format(i, cost))
    return W, b, J_history


def main():
    X, Y_target, m, dataframe = get_data()


    plot_data(dataframe,0,0, mode=0)

    W = np.zeros(2)  # shape = (1,2) or (2,)
    b = 0
    theta = np.append(W,b)
    # print(theta)

    W, b, *rest = run_gradient_descent(X, Y_target, m, W, b, lr=0.1, iters=20000)

    # Y_predict = np.zeros_like(m,1)
    # arguments: theta, *args
    # result = opt.fmin_tnc(func=cost_function, x0=theta, fprime=compute_gradient_descent, args=(X, Y_target,
    #                                                                                            Y_predict,m,W,b))
    # print(result)

    W = np.matrix(W)
    plot_data(dataframe, W, b, mode=1)

    # print("shapeXa", X[:, 0].reshape(m,).shape)
    # a = np.arange(1,5,1)
    # b = np.arange(3,10,2)
    # c = a * b  # just multiple one on one
    # d = np.dot(a, b)  # multiple and add up
    # print(a,b,c,d)

    
if __name__ == '__main__':
    main()

    