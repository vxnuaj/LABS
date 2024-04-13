# ROUND 2, IMPLEMENTED WITH PURE FIRST PRINCIPLES THINKING. ** besides an isuse w sigmoid lol ** 

import numpy as np
import pandas as pd

def init_params():
    w = np.random.rand(2, 1)
    b = np.random.rand(1,1)
    return w, b

def forward(x, w, b):
    z = np.dot(x, w) + b
    a = sigmoid(z)
    return a

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y, a):
    episilon = 1e-10
    loss = np.mean(-y * np.log(a + episilon) - (1-y) * np.log(1-a + episilon))
    return loss


def gradients(w, b, a, y, x, alpha):
    dw = np.mean((a-y) * x)
    db = np.mean(a - y)
    w = w - alpha * dw
    b = b - alpha * db
    return w, b


def gradient_descent(x, y, alpha, epochs):
    w, b = init_params()

    for epoch in range(epochs):
        a = forward(x, w, b)
        w, b = gradients(w, b, a, y, x, alpha)
        loss = log_loss(y ,a)

        print(f"Epoch: {epoch}")
        print(f"Loss: {loss}")

    return w, b


if __name__ == "__main__":

    data = pd.read_csv("Artificial-Intelligence/Machine-Learning/Logistic-Regression/Data/randomtrain.csv")
    data = np.array(data)

    X_train = data[:, 0:2]
    Y_train = data[:, 2].reshape(-1,1)

    alpha = .0001
    epochs = 5000

    gradient_descent(X_train, Y_train, alpha, epochs)

