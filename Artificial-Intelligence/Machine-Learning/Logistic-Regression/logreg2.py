import numpy as np
import pandas as pd

def init_params():
    w = np.random.rand(2, 1)
    b = np.random.rand(1, 1)
    return w, b

def forward(x, w, b):
    z = np.dot(x, w) + b
    a = sigmoid(z)
    return a

def sigmoid(z):
    return 1 / (1-np.exp(-z))

def log_loss(a, y):
    loss = np.mean(- y * np.log(a) - (1-y) * np.log(1-a))
    return loss

def back_prop(a, y, x, alpha, w, b):
    dw = np.mean((a - y) * x) #gradient of bias wrt loss
    db = np.mean(a - y)#gradient of bias wrt loss
    w = w - alpha * dw #update rule
    b = b - alpha * db #update rule
    return w, b


def gradient_descent(x, y, alpha, epochs):
    w, b = init_params()

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")

        a = forward(x, w, b)
        loss = log_loss(a, y)
        w, b = back_prop(a, y, x, alpha, w, b)

        print(f"Loss: {loss}")

    return w, b


if __name__ == "__main__":

    data = pd.read_csv("./Data/randomtrain.csv")
    data = np.array(data)
    x = data[:, 0:2]
    y = data[:, 2].reshape(-1,1)
    alpha = .0001
    epochs = 1000


    w, b = gradient_descent(x, y, alpha, epochs)

    print(f"Weight: {w}")
    print(f"Bias: {b}")
