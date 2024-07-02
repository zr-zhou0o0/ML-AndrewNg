import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('.../deeplearning.mplstyle')
plt.ion() # interactive mode

def plot(x_tr, y_tr, f):
    plt.title('Profits for a Food Truck')
    plt.ylabel('Profits')  # set_ylabel has an object but ylabel has not
    plt.xlabel('Population of the city')
    plt.scatter(x_tr, y_tr, marker='x', c='r')
    plt.plot(x_tr, f, label='linear')
    plt.show()
    plt.savefig("linear_regression")

def model_function(x, w, b):
    m = x.shape[0]
    f = np.zeros(m)
    for i in range(m):
        # print("fi={},xi={}".format(f[i],x[i]))
        f[i] = x[i] * w + b
    return f

def compute_cost(x_tr, y_tr, w, b):
    m = x_tr.shape[0]
    y_predict = np.zeros(m)
    square = np.zeros(m)
    for i in range(m):
        y_predict[i] = x_tr[i] * w + b
        square[i] = (y_predict[i] - y_tr[i])**2
    cost = np.sum(square)/(2*m)
    return cost

def compute_gradient(x_tr, y_tr, w, b):
    m = x_tr.shape[0]
    y_predict = np.zeros(m)
    dif = np.zeros(m)
    dif1 = np.zeros(m)
    for i in range(m):
        y_predict[i] = x_tr[i] * w + b
        dif[i] = y_predict[i] - y_tr[i]
        dif1[i] = dif[i] * x_tr[i]
    derivative_w = np.sum(dif1)/m
    derivative_b = np.sum(dif)/m
    return derivative_w, derivative_b

def gradient_descent(x_tr, y_tr, w, b, lr):
    derivative_w, derivative_b = compute_gradient(x_tr, y_tr, w, b)
    w_temp = w - lr * derivative_w
    b_temp = b - lr * derivative_b
    return w_temp, b_temp


def main():
    # get data
    path = 'ex1data1.txt'
    dataframe_org = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    # print(dataframe_org.head)
    # print(dataframe_org.describe())

    # delete the misdata
    # bug: after delete the misdata, the index remains unchanged. 1 to 96 with 93 data.
    # reset index
    dataframe=dataframe_org[dataframe_org['Profit'] > 0]
    dataframe.reset_index(drop=True, inplace=True)
    # print(dataframe)
    # print(dataframe.describe())

    x_train = dataframe['Population']
    y_train = dataframe['Profit']
    w = 0
    b = 0
    m = x_train.shape[0]
    f = np.zeros(m)

    # plot the data
    f = model_function(x_train, w, b)
    plot(x_train, y_train, f)

    # learning algorithm
    cost = 0
    lr = 0.01 # if learning rate equals to 0.1, it will diverge
    print("here")
    for i in range(500):
        w_temp, b_temp = gradient_descent(x_train, y_train, w, b, lr)
        w = w_temp
        b = b_temp
        cost = compute_cost(x_train, y_train, w, b)
        print(cost)

    # plot
    f = model_function(x_train, w, b)
    plot(x_train, y_train, f)


if __name__ == '__main__':
    main()