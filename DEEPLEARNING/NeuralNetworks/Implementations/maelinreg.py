'''

Linear Regression with Mean Absolute Error

'''

import numpy as np
import pandas as pd

def init_params():
    w = np.random.randn(1,1)
    b = np.zeros((1,1))
    return w, b

def forward(x, w, b):
    pred = np.dot(w, x) + b
    return pred

def mae(y, pred):
    l = np.mean(np.abs((y - pred)))
    return l

def mae_grad(y, pred):
    return - np.sign(y - pred)

def backwards(x, y, pred):
    dz = mae_grad(y, pred)
    dw = np.dot(dz, x.T) / x.shape[1]
    db = np.sum(dz) / x.shape[1]
    return dw, db

def update(w, b, dw, db, alpha):
    w = w - alpha * dw
    b = b - alpha * db
    return w, b

def grad_descent(x, y, w, b, epochs, alpha):
    for epoch in range(epochs):
        pred = forward(x, w, b)

        l = mae(y, pred)

        dw, db = backwards(x, y, pred)
        w, b = update(w, b, dw, db, alpha)

        if epoch % 1000 == 0:
            print(f"epoch: {epoch}")
            print(f"mse: {l}")

    print(f"slope: {w}")
    print(f"intercept: {b}")
    
    return w, b

def model(x, y, epochs, alpha):
    w, b = init_params()
    w, b = grad_descent(x, y, w, b, epochs, alpha)
    return w, b

if __name__ == "__main__":

    data = pd.read_csv('data/random1.csv')
    data = np.array(data)

    X_train = data[:, :1].T
    Y_train = data[:, 1].reshape(1, -1)

    w, b = model(X_train, Y_train, 500000, .001)