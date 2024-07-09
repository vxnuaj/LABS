import numpy as np
import pandas as pd
import pickle

def load_model(file):
    with open(file) as f:
        return pickle.load(f)

def save_model(file, w1, b1, w2, b2, w3, b3):
    with open(file) as f:
        pickle.dump((w1, b1, w2, b2, w3, b3), f)

def init_params():
    rng = np.random.default_rng(seed = 1)
    w1 = rng.normal(size = (32, 784)) * np.sqrt(2/784)
    b1 = np.zeros((32, 1))
    w2 = rng.normal(size = (10 , 32)) * np.sqrt(2/32)
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def leaky_relu(z):
    return np.where(z > 0, z, (.01 * z))

def leaky_relu_deriv(z):
    return np.where(z>0, 1, .01)

def softmax(z):
    eps = 1e-6
    return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims=True)

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(y):
    one_hot_y = np.zeros((np.max(y)+1, y.size)) #10, 60000
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y

def cat_cross(one_hot_y, a):
    eps = 1e-8
    l = -np.sum(one_hot_y * np.log(a + eps)) / 60000
    return l

def accuracy(y, a2):
    pred = np.argmax(a2, axis = 0)
    acc = np.sum(pred == y) / 60000 * 100
    return acc


def backward(x, one_hot_y, w2, a2, a1, z2, z1):
    dz2 = a2 - one_hot_y # difference between a one_hot_y vs regular y?
    dw2 = np.dot(dz2, a1.T) / 60000
    db2 = np.sum(dz2, axis = 1, keepdims=True) / 60000
    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1)
    dw1 = np.dot(dz1, x.T) / 60000
    db1 = np.sum(dz1, axis=1, keepdims=True) / 60000
    return dw2, db2, dw1, db1

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
        acc = accuracy(y, a2)

        dw2, db2, dw1, db1 = backward(x, one_hot_y, w2, a2, a1, z2, z1)
        w1, b1, w2, b2 = update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        print(f"epoch: {epoch}")
        print(f"loss: {l}")
        print(f"acc: {acc}%\n")

    save_model(file, w1, b1, w2, b2)
    return w1, b1, w2, b2
    
def model(x, y, epochs, alpha, file):
    try:
        w1, b1, w2, b2 = load_model(file)
        print("found model! initializing model params!")

    except FileNotFoundError:
        print("model not found! initializing new params!")
        w1, b1, w2, b2= init_params()
    
    w1, b1, w2, b2 = gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha, file)
    return w1, b1, w2, b2

if __name__ == "__main__":

    data = np.array(pd.read_csv('data/fashion-mnist_train.csv'))

    X_train = data[:, 1:786].T / 255 #784, 60000
    Y_train = data[:, 0].reshape(1, -1) #1, 60000

    file = 'models/mnistnn.pkl'

    w1, b1, w2, b2 = model(X_train, Y_train, 1000, .1, file)
    
'''
reflection.

april: 12a deeper network trains slower. for the mnist classification task, i think less layers makes sense. maybe because it lacks complexity?

may 12: wtf, tanh trains my model like 2x faster. steeper gradient ig?
'''