import numpy as np
import pandas as pd

def init_params():
    w = np.random.randn(1, 13) * np.sqrt(1/13)
    b = np.zeros((1, 1))
    return w, b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(x, w, b):
    z = np.dot(w, x) + b
    a = sigmoid(z)
    return z, a

def accuracy(a, y):
    pred = np.round(a, decimals = 0)
    acc = np.sum(pred == y) / 1025 * 100
    return acc

def bce(y, a):
    eps = 1e-10
    l = - np.sum(y * np.log(a + eps) + (1 - y) * np.log(1 - a + eps)) / 1025
    return l

def backwards(x, y, a):
    dz = a - y
    dw = np.dot(dz, x.T) / 1025
    db = np.sum(dz) / 1025
    return dw, db

def update(w, b, dw, db, alpha):
    w = w - alpha * dw
    b = b - alpha * db
    return w, b

def grad_descent(x, y, w, b, epochs, alpha):
    for epoch in range(epochs):
        z, a = forward(x, w, b)

        l = bce(y, a)
        acc = accuracy(a, y)

        dw, db = backwards(x, y ,a)
        w, b = update(w, b, dw, db, alpha)

        if epoch % 1000 == 0:
            print(f"epoch: {epoch}")
            print(f"acc: {acc:.3f}%")
            print(f"loss: {l}\n")
    return w, b

def model(x, y, epochs, alpha):
    w, b = init_params()
    w, b = grad_descent(x, y, w, b, epochs, alpha)
    return w, b


if __name__ == "__main__":
    data = pd.read_csv('data/heart.csv')
    data = np.array(data)

    X_train = data[:, 0:13].T
    Y_train = data[:, 13].reshape(1, -1)

    w, b = model(X_train, Y_train, 100000, .0003)