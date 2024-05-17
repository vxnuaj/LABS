'''

A Neural Network on the MNIST datset, using tanh/relu/selu/swish/elu, softmax, and categorical cross entropy

'''

import numpy as np
import pandas as pd
import activfuncs as af

def init_params():
    w1 = np.random.randn(32, 784) * np.sqrt(1 / 784)
    b1 = np.zeros((32, 1))
    w2 = np.random.randn(10 ,32) * np.sqrt(1 / 784)
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def tanh(z):
    eps = 1e-10
    return (np.exp(z + eps) - np.exp(-z + eps)) / (np.exp(z + eps) + np.exp(-z + eps))

def tanh_deriv(z):
    return 1 - (tanh(z) ** 2)

def softmax(z):
    eps = 1e-10
    return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims = True)

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = af.elu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(y):
    one_hot_y = np.zeros(((np.max(y) + 1), y.size))
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y

def cce(one_hot_y, a):
    eps = 1e-10
    l = - np.sum(one_hot_y * np.log(a + eps)) / 60000
    return l

def acc(y, a2):
    pred = np.argmax(a2, axis=0, keepdims=True)
    acc = np.sum(y == pred) / y.size * 100
    return acc

def backward(x, one_hot_y, w2, a2, a1, z1):
    dz2 = a2 - one_hot_y
    dw2 = np.dot(dz2, a1.T) / 60000
    db2 = np.sum(dz2, axis = 1, keepdims=True) / 60000
    dz1 = np.dot(w2.T, dz2) * af.elu_deriv(z1)
    dw1 = np.dot(dz1, x.T) / 60000
    db1 = np.sum(dz1, axis = 1, keepdims = True) / 60000
    return dw1, db1, dw2, db2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2
    
def grad_descent(x, y, w1, b1, w2, b2, alpha, epochs ):
    one_hot_y = one_hot(y)
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)

        l = cce(one_hot_y, a2)
        accuracy = acc(y, a2)

        dw1, db1, dw2, db2 = backward(x, one_hot_y, w2, a2, a1, z1)
        w1, b1, w2, b2 = update(w1, b1 ,w2, b2, dw1, db1, dw2, db2, alpha)

        print(f"Epoch: {epoch}")
        print(f"Acc: {accuracy}")
        print(f"Loss: {l}")

    return w1, b1, w2, b2

def model(x, y, alpha, epochs):
    w1, b1, w2, b2 = init_params()
    w1, b1, w2, b2 = grad_descent(x, y, w1, b1, w2, b2, alpha, epochs)
    return w1, b1, w2, b2


if __name__ == "__main__":
    
    data = pd.read_csv('data/mnist_train.csv')

    data = np.array(data)

    X_train = data[:, 1:786].T /255
    Y_train = data[:, 0].reshape(1, -1)

    model(X_train, Y_train, .1, 1000)
