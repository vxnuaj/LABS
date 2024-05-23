'''

L1 regularization of Linear Regression

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

def mae(y, pred, w, lambd):
    manhnorm = (lambd * np.sum(np.abs(w))) / y.size
    l = np.mean(np.abs((y - pred)))
    l = l+manhnorm
    return manhnorm, l

def mae_grad(y, pred):
    return - np.sign(y - pred)

def backwards(x, y, w, pred, lambd):
    dz = mae_grad(y, pred)
    dw = (np.dot(dz, x.T)) / y.size + (lambd * np.sign(w) / y.size)
    db = np.sum(dz) / y.size
    return dw, db

def update(w, b, dw, db, alpha):
    w = w - alpha * dw
    b = b - alpha * db
    return w, b

def grad_descent(x, y, w, b, epochs, alpha, lambd):
    for epoch in range(epochs):
        pred = forward(x, w, b)

        manhnorm, l = mae(y, pred, w, lambd)

        dw, db = backwards(x, y, w, pred, lambd)
        w, b = update(w, b, dw, db, alpha)

        if epoch % 1000 == 0:
            print(f"epoch: {epoch}")
            print(f"mse: {l}")

    print(f"slope: {w}")
    print(f"intercept: {b}")
    
    return w, b

def model(x, y, epochs, alpha, lambd):
    w, b = init_params()
    w, b = grad_descent(x, y, w, b, epochs, alpha, lambd)
    return w, b

if __name__ == "__main__":

    data = pd.read_csv('data/toy.csv')
    data = np.array(data)

    X_train = data[:, :1].T
    Y_train = data[:, 1].reshape(1, -1)

    w, b = model(X_train, Y_train, 500000, .001, 100)