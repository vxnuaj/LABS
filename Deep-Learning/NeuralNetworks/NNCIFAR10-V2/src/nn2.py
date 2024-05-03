''''
spedran this, 30 min?
'''

import numpy as np
import pickle
from preprocess import X_train, Y_train

def save_model(file, w1, b1, w2, b2):
    with open(file, 'wb') as f:
        pickle.dump((w1, b1, w2, b2), f)

def load_model(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
 
def init_params():
    w1 = np.random.randn(64, 3072) * np.sqrt(1/3072)
    b1 = np.zeros((64, 1))
    w2 = np.random.randn(10, 64) * np.sqrt(1/3072)
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def relu(z):
    return np.maximum(z, 0)

def relu_deriv(z):
    return z > 0

def softmax(z):
    eps = 1e-10
    return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims=True)

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2) # 10, 10000
    return z1, a1, z2, a2

def one_hot(y):
    one_hot_y = np.zeros((np.max(y) + 1, y.size))
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y

def cat_cross(one_hot_y, a):
    eps = 1e-10
    l = - np.sum(one_hot_y * np.log(a + eps)) / 10000
    return l

def accuracy(a2, y):
    pred = np.argmax(a2, axis = 0, keepdims=True)
    acc = np.sum(y == pred) / 10000 * 100
    return acc

def backward(x, one_hot_y, a2, a1, w2):
    dz2 = a2 - one_hot_y
    dw2 = np.dot(dz2, a1.T) / 10000
    db2 = np.sum(dz2, axis = 1, keepdims = True) / 10000
    dz1 = np.dot(w2.T, dz2) * relu_deriv(a1)
    dw1 = np.dot(dz1, x.T) / 10000
    db1 = np.sum(dz1, axis = 1, keepdims = True) / 10000
    return dw1, db1, dw2, db2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha, file):
    one_hot_y = one_hot(y)
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)

        l = cat_cross(one_hot_y, a2)
        acc = accuracy(a2, y)

        dw1, db1, dw2, db2 = backward(x, one_hot_y, a2, a1, w2)
        w1, b1, w2, b2 = update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        print(f"epoch: {epoch}")
        print(f"loss: {l}")
        print(f"acc: {acc}% \n")

    save_model(file)
    return w1, b1, w2, b2

def model(x, y, alpha, epochs, file):
    try:
        load_model(file)
        print("loading modl!")
    except FileNotFoundError:
        print("model not found, initializing new model!")
        w1, b1, w2, b2 = init_params()
    w1, b1, w2, b2 = gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha, file)
    return w1, b1, w2, b2



if __name__ == "__main__":
    w1, b1, w2, b2 = model(X_train, Y_train, .01, 1000, 'model/nn2.pkl')




