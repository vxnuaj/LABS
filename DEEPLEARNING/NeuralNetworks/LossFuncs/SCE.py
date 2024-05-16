import numpy as np
import pandas as pd

def init_params():
    w1 = np.random.randn(32, 784) * np.sqrt(1/784)
    b1 = np.zeros((32, 1))
    w2 = np.random.randn(10, 32) * np.sqrt(1/784)
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def leaky_relu(z):
    return np.where(z > 0, z, .01 * z)

def leaky_relu_deriv(z):
    return np.where(z>0, 1, .01)

def softmax(z):
    eps = 1e-10
    return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims=True)

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2) #10, 60000
    return z1, a1, z2, a2

def accuracy(a, y):
    pred = np.argmax(a, axis = 0)
    acc = np.sum(pred == y) / 60000 * 100
    return acc

def one_hot(y):
    one_hot_y = np.zeros((np.max(y) + 1, y.size ))
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y

def smooth_hot(one_hot_y, alpha, classes):
    smth_hot = one_hot_y * (1 - alpha) + (alpha / classes)
    return smth_hot

def loss(smth_hot, a):
    eps = 1e-10
    l = - np.sum(smth_hot * np.log(a + eps)) / 60000
    return l

def backward(x, smth_hot, w2, a2, a1, z1):
    dz2 = a2 - smth_hot
    dw2 = np.dot(dz2, a1.T) / 60000
    db2 = np.sum(dz2, axis = 1, keepdims=True) / 60000
    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1)
    dw1 = np.dot(dz1, x.T) / 60000
    db1 = np.sum(dz1, axis = 1, keepdims = True) / 60000
    return dw1, db1, dw2, db2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha):
    one_hot_y = one_hot(y)
    smth_hot = smooth_hot(one_hot_y, .05, 10)
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)

        l = loss(smth_hot, a2)
        acc = accuracy(a2, y)

        dw1, db1, dw2, db2 = backward(x, smth_hot, w2, a2, a1, z1)
        w1, b1, w2, b2 = update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        
        print(f"epoch: {epoch}")
        print(f"acc: {acc}")
        print(f"Loss: {l}\n")

    return w1, b1, w2, b2

def model(x, y, epochs, alpha):
    w1, b1, w2, b2 = init_params()
    w1, b1, w2, b2 = gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha)
    return w1, b1, w2, b2

if __name__ == "__main__":
    data = pd.read_csv('data/fashion-mnist_train.csv') # 900 samples, 7 Features

    data = np.array(data) #60000, 785

    X_train = data[:, 1:785].T / 255 #784, 60000
    Y_train = data[:, 0].reshape(1, -1) #1, 60000

    model(X_train, Y_train, 1000, .1)