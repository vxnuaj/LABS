import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle



'''
2 hidden layers, one output layer

h1 = 64 neurons
h2 = 32 neurons
output layer = 10 neurons

qs: should i write the env i'm in on the top of the script? ans: not needed

'''

def load_model(file):
    with open(file) as f:
        return pickle.load(f)

def save_model(file, w1, b1, w2, b2, w3, b3):
    with open(file) as f:
        pickle.dump((w1, b1, w2, b2, w3, b3), f)

def init_params():
    w1 = np.random.randn(32, 784) * np.sqrt(1/784)
    b1 = np.zeros((32, 1))
    w2 = np.random.randn(16 , 32) * np.sqrt(1/784)
    b2 = np.zeros((16, 1))
    w3 = np.random.randn(10, 16) * np.sqrt(1/784)
    b3 = np.zeros((10, 1))
    return w1, b1, w2, b2, w3, b3

def relu(z):
    return np.maximum(z, 0)

def relu_deriv(z):
    return z > 0


def softmax(z):
    eps = 1e-6
    return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims=True)

def forward(x, w1, b1, w2, b2, w3, b3):
    z1 = np.dot(w1, x) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = softmax(z3) #10, 60000
    return z1, a1, z2, a2, z3, a3

def one_hot(y):
    one_hot_y = np.zeros((np.max(y)+1, y.size)) #10, 60000
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y

def cat_cross(one_hot_y, a):
    eps = 1e-10
    l = -np.sum(one_hot_y * np.log(a + eps)) / 60000
    return l

def accuracy(y, a3):
    pred = np.argmax(a3, axis = 0)
    acc = np.sum(pred == y) / 60000 * 100
    return acc


def backward(x, one_hot_y, w3, w2, a3, a2, a1, z2, z1):
    dz3 = a3 - one_hot_y # difference between a one_hot_y vs regular y?
    dw3 = np.dot(dz3, a2.T) / 60000
    db3 = np.sum(dz3, axis = 1, keepdims = True) / 60000
    dz2 = np.dot(w3.T, dz3) * relu_deriv(z2)
    dw2 = np.dot(dz2, a1.T) / 60000
    db2 = np.sum(dz2, axis = 1, keepdims=True) / 60000
    dz1 = np.dot(w2.T, dz2) * relu_deriv(z1)
    dw1 = np.dot(dz1, x.T) / 60000
    db1 = np.sum(dz1, axis=1, keepdims=True) / 60000
    return dw3, db3, dw2, db2, dw1, db1

def update(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    w3 = w3 - alpha * dw3
    b3 - b3 - alpha * db3
    return w1, b1, w2, b2, w3, b3

def gradient_descent(x, y, w1, b1, w2, b2, w3, b3, epochs, alpha, file):
    one_hot_y = one_hot(y)
    for epoch in range(epochs):
        z1, a1, z2, a2, z3, a3 = forward(x, w1, b1, w2, b2, w3, b3)

        l = cat_cross(one_hot_y, a3)
        acc = accuracy(y, a3)

        dw3, db3, dw2, db2, dw1, db1 = backward(x, one_hot_y, w3, w2, a3, a2, a1, z2, z1)
        w1, b1, w2, b2, w3, b3 = update(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha)

        print(f"epoch: {epoch}")
        print(f"loss: {l}")
        print(f"acc: {acc}%\n")

    save_model(file, w1, b1, w2, b2, w3, b3)
    return w1, b1, w2, b2, w3, b3\
    
def model(x, y, epochs, alpha, file):
    try:
        w1, b1, w2, b2, w3, b3 = load_model(file)
        print("found model! initializing model params!")

    except FileNotFoundError:
        print("model not found! initializing new params!")
        w1, b1, w2, b2, w3, b3 = init_params()
    
    w1, b1, w2, b2, w3, b3 = gradient_descent(x, y, w1, b1, w2, b2, w3, b3, epochs, alpha, file)
    return w1, b1, w2, b2, w3, b3

if __name__ == "__main__":

    data = np.array(pd.read_csv('data/mnist_train.csv'))

    X_train = data[:, 1:786].T / 255 #784, 60000
    Y_train = data[:, 0].reshape(1, -1) #1, 60000

    file = 'models/mnistnn.pkl'

    w1, b1, w2, b2, w3, b3 = model(X_train, Y_train, 1000, .01, file)
    
'''
reflection.

a deeper network trains slower. for the mnist classification task, i think less layers makes sense. maybe because it lacks complexity?
'''