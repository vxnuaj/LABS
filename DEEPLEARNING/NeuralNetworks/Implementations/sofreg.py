'''

Softmax Regression / Multinomial Logistic Regression

'''


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def init_params():
    w = np.random.randn(3, 4)
    b = np.zeros((3,1))
    return w, b

def softmax(z):
    z = z.astype(float)
    return np.exp(z) / np.sum(np.exp(z), axis = 0, keepdims = True)

def forward(x, w, b):
    z = np.dot(w, x) + b
    a = softmax(z)
    return a # 3, 150

def one_hot(y):
    one_hot_y = np.zeros((np.max(y) + 1, y.size))
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y

def cce(one_hot_y, a):
    l = - np.sum(one_hot_y * np.log(a)) / 150
    return l

def accuracy(y, a):
    pred = np.argmax(a, axis = 0, keepdims=True)
    acc = np.sum(y == pred) / 150 * 100
    return acc

def backwards(x, one_hot_y, a):
    dz = a - one_hot_y
    dw = np.dot(dz, x.T)
    db = np.sum(dz, axis = 1, keepdims = True)
    return dw, db

def update(w, b, dw, db, alpha):
    w = w - alpha * dw
    b = b - alpha * db
    return w, b

def gradient_descent(x, y, w, b, alpha, epochs):
    one_hot_y = one_hot(y)
    for epoch in range(epochs):
        a = forward(x, w, b)

        l = cce(one_hot_y, a)
        acc = accuracy(y, a)

        dw, db = backwards(x, one_hot_y, a)
        w, b = update(w, b, dw, db, alpha)

        if epoch % 1000 == 0:
            print(f"epoch: {epoch}")
            print(f"acc: {acc}")
            print(f"loss: {l}\n")

    return w, b

def model(x, y, alpha, epochs):
    w, b = init_params()
    w, b = gradient_descent(x, y, w, b, alpha, epochs)
    return w, b


if __name__ == "__main__":
    data = pd.read_csv('data/iris.csv')
    data = np.array(data)

    X_train = data[:, 0:4].T
    Y_train = data[:, 4]

    le = LabelEncoder()

    Y_train = le.fit_transform(Y_train).reshape(1, -1)


    w, b = model(X_train, Y_train, .001, 10000)