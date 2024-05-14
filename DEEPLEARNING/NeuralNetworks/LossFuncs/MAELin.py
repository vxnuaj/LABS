

#implement lin reg with random1.csv with MAE

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def init_param():
    w = np.zeros((1,200))
    b = np.random.randn(1, 1)
    return w, b

def forward(x, w, b):
    pred = np.dot(w, x) + b
    return pred

def mae(y, pred):
    l = (1 / 200) * np.mean(np.abs(y - pred))
    return l

def backward(x, y, pred):
    dz = -np.sign(y - pred)
    dw = np.dot(dz, x.T) / 200
    db = np.sum(dz, axis = 0, keepdims = True) / 200
    return dw, db

def update(w, b, dw, db, alpha):
    w = w - alpha * dw
    b = b - alpha * db
    return w, b

def gradient_descent(x, y, w, b, epochs, alpha):
    for epoch in range(epochs):
        pred = forward(x, w, b)

        l = mae(y, pred)

        dw, db = backward(x, y, pred)
        w, b = update(w, b, dw, db, alpha)


        print(f"Epoch: {epoch}")
        print(f"Loss: {l}")

    return w, b

def model(x, y, epochs, alpha):
    w, b = init_param()
    w, b = gradient_descent(x, y, w, b, epochs, alpha)
    return w, b

if __name__ == "__main__":
    data = pd.read_csv('data/tvmarketing.csv')
    data = np.array(data)

    X_train = data[:, 0].reshape(-1, 1) # 200, 1
    Y_train = data[:, 1].reshape(-1, 1) # 200, 1

    ss = StandardScaler()
    sw = StandardScaler()

    X_train = data[:, 0].reshape(-1, 1)
    ss.fit(X_train)
    X_train = ss.transform(X_train)

    Y_train = data[:, 1].reshape(-1, 1) # 200, 1
    sw.fit(Y_train)
    Y_train = sw.transform(Y_train)

    model(X_train, Y_train, 1000, .1)