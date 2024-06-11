import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def load_model(file):
    with open(file) as f:
        return pickle.load(f)

def save_model(file, w1, b1, w2, b2, w3, b3):
    with open(file) as f:
        pickle.dump((w1, b1, w2, b2, w3, b3), f)

def init_params():
    rng = np.random.default_rng(seed = 1)
    w1 = rng.normal(size = (32, 784)) * np.sqrt(1/784)
    b1 = np.zeros((32, 1))
    w2 = rng.normal(size = (10, 32)) * np.sqrt(1/784)
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def leaky_relu(z):
    return np.where(z > 0, z, (.01 * z))

def leaky_relu_deriv(z):
    return np.where(z>0, 1, .1)

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

def cat_cross(one_hot_y, a, w2, w1, lambd):
    eps = 1e-10
    fronorm = ((lambd * np.sum(np.square(w2))) + (lambd * np.sum(np.square(w1)))) / 60000
    l = -np.sum(one_hot_y * np.log(a + eps)) / 60000    
    reg_l = l + fronorm
    return reg_l

def accuracy(y, a2):
    pred = np.argmax(a2, axis = 0)
    acc = np.sum(pred == y) / 60000 * 100
    return acc


def backward(x, one_hot_y, w2, w1, a2, a1, z2, z1, lambd):
    dz2 = a2 - one_hot_y
    dw2 = (np.dot(dz2, a1.T) + (2 * lambd * w2)) / 60000
    db2 = np.sum(dz2, axis = 1, keepdims=True) / 60000
    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1)
    dw1 = (np.dot(dz1, x.T) + (2 * lambd * w1)) / 60000
    db1 = np.sum(dz1, axis=1, keepdims=True) / 60000
    return dw2, db2, dw1, db1

def rms(beta, dw1, db1, dw2, db2, vdw1_p, vdb1_p, vdw2_p, vdb2_p):
    vdw1 = (beta * vdw1_p) + ((1 - beta) * np.square(dw1))
    vdb1 = (beta * vdb1_p) + ((1 - beta) * np.square(db1))
    vdw2 = (beta * vdw2_p) + ((1 - beta) * np.square(dw2))
    vdb2 = (beta * vdb2_p) + ((1 - beta) * np.square(db2))
    return vdw1, vdb1, vdw2, vdb2

def update_rms(w1, b1, w2, b2, dw1, db1, dw2, db2, vdw1, vdb1, vdw2, vdb2, alpha):
    eps = 1e-8
    w1 = w1 - alpha * (dw1 / np.sqrt(vdw1 + eps))
    b1 = b1 - alpha * (db1 / np.sqrt(vdb1 + eps))
    w2 = w2 - alpha * (dw2 / np.sqrt(vdw2 + eps))
    b2 = b2 - alpha * (db2 / np.sqrt(vdb2 + eps))
    return w1, b1, w2, b2

def gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha, beta, lambd, file):
    one_hot_y = one_hot(y)

    #Instantiating initial values for RMSprop as 0

    vdw1 = 0
    vdb1 = 0
    vdw2 = 0
    vdb2 = 0

    epochs_vec = []
    acc_vec = []
    loss_vec = []

    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)

        l = cat_cross(one_hot_y, a2, w2, w1, lambd)
        acc = accuracy(y, a2)

        dw2, db2, dw1, db1 = backward(x, one_hot_y, w2, w1, a2, a1, z2, z1, lambd)

        vdw1, vdb1, vdw2, vdb2 = rms(beta, dw1, db1, dw2, db2, vdw1, vdb1, vdw2, vdb2)

        w1, b1, w2, b2 = update_rms(w1, b1, w2, b2, dw1, db1, dw2, db2, vdw1, vdb1, vdw2, vdb2, alpha)

        print(f"epoch: {epoch}")
        print(f"loss: {l}")
        print(f"acc: {acc}%")
        print(f"vdw2: {np.max(vdw2)}")

        epochs_vec.append(epoch)
        loss_vec.append(l)
        acc_vec.append(acc)

    return w1, b1, w2, b2, epochs_vec, loss_vec, acc_vec
    
def model(x, y, epochs, alpha, beta, lambd, file):
    try:
        w1, b1, w2, b2 = load_model(file)
        print("found model! initializing model params!")

    except FileNotFoundError:
        print("model not found! initializing new params!")
        w1, b1, w2, b2= init_params()
    
    w1, b1, w2, b2, epochs_vec, loss_vec, acc_vec = gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha, beta, lambd, file)
    return w1, b1, w2, b2, epochs_vec, loss_vec, acc_vec

if __name__ == "__main__":

    data = np.array(pd.read_csv('data/fashion-mnist_train.csv'))

    X_train = data[:, 1:786].T / 255 #784, 60000
    Y_train = data[:, 0].reshape(1, -1) #1, 60000

    file = '../models/BatchNN.pkl'

    w1, b1, w2, b2, epochs_vec, loss_vec, acc_vec= model(X_train, Y_train, epochs = 250, alpha = .001, beta = .9, lambd = 10, file = file)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    axs[0].plot(epochs_vec, acc_vec, label='Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    axs[1].plot(epochs_vec, loss_vec, label='Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.show()