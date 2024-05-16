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
    eps = 1e-9  # Small constant to avoid division by zero
    z -= np.max(z, axis=0)  # Subtract the maximum value for each sample
    return np.exp(z + eps) / np.sum(np.exp(z + eps), axis=0, keepdims=True)

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2) #10, 60000
    return z1, a1, z2, a2

def predict(a):
    pred = np.argmax(a, axis = 0)
    pred = pred.reshape(1, -1)
    return pred

def accuracy(y, pred):
    acc = np.sum(pred == y) / 60000 * 100
    return acc

def loss(y, a2):
    eps = 1e-10
    l = - np.sum(y * np.log(a2 + eps)) / 60000
    return l

def backward(x, y, w2, a2, a1, z1):
    dz2 = a2 - y
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
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)

        pred = predict(a2)
        acc = accuracy(y, pred)
        l = loss(y, a2)

        dw1, db1, dw2, db2 = backward(x, y, w2, a2, a1, z1)
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

    model(X_train, Y_train, 1000, .01)

'''    w1, b1, w2, b2 = init_params()
    z1, a1, z2, a2 = forward(X_train, w1, b1, w2, b2)

    acc = accuracy(a2, Y_train)
    pred = predict(a2)

'''
