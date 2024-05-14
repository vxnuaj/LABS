import numpy as np
import pandas as pd
import sklearn.preprocessing as spp

def binary_num(y):
    Y_labels = []
    for label in y:
        if label == 'Kecimen':
            Y_labels.append(0)
        elif label == 'Besni':
            Y_labels.append(1)
    return np.array(Y_labels)

def init_params():
    w1 = np.random.randn(3, 7)
    b1 = np.zeros((3, 1))
    w2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    return w1, b1, w2, b2

def leaky_relu(z):
    return np.where(z > 0, z, .01 * z)

def leaky_relu_deriv(z):
    return np.where(z > 0, 1, .01)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def loss(y, a):
    eps = 1e-10
    l = -(1 / 900) * np.sum(y * np.log(a + eps) + (1 - y) * np.log(1 - a + eps))
    return l

def backward(x, y, w2, a2, a1, z1):
    dz2 = a2 - y
    dw2 = np.dot(dz2, a1.T) / 900
    db2 = np.sum(dz2, axis = 1, keepdims=True)
    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1)
    dw1 = np.dot(dz1, x.T) / 900
    db1 = np.sum(dz1) / 900
    return dw1, db1, dw2, db2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    return w1, b1, w2, b2

def gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha):
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)

        l = loss(y, a2)

        dw1, db1, dw2, db2 = backward(x, y, w2, a2, a1, z1)
        w1, b1, w2, b2 = update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        
        if epoch % 1000 == 0:
            print(f"epoch: {epoch}")
            print(f"Loss: {l}")

    return w1, b1, w2, b2

def model(x, y, epochs, alpha):
    w1, b1, w2, b2 = init_params()
    w1, b1, w2, b2 = gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha)
    return w1, b1, w2, b2

if __name__ == "__main__":
    data = pd.read_csv('data/Raisin_Dataset.csv') # 900 samples, 7 Features
    X_train = data.iloc[:, :7]
    Y_train = data.iloc[:, 7]
    scalar = spp.MinMaxScaler()
    X_train = scalar.fit_transform(X_train)
    X_train = np.array(X_train).T
    Y_train = np.array(Y_train)
    Y_train = binary_num(Y_train).reshape(1, -1)

    model(X_train, Y_train, 100000, .1)

