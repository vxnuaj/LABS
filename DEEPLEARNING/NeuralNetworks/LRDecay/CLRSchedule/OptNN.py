'''

Implementing the cyclical learning rate in accordance with the min / max values found in OptLR.py

https://arxiv.org/pdf/1506.01186

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def save_model(file, w1, b1, w2, b2):
    with open(file, 'wb') as f:
        pickle.dump((w1, b1, w2, b2), f)

def load_model(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def init_params():

    rng = np.random.default_rng(seed = 1)

    w1 = rng.normal(size = (64, 784)) * np.sqrt(1/784)
    b1 = np.zeros((64, 1))
    w2 = rng.normal(size = (10, 64)) * np.sqrt(1/784)
    b2 = np.zeros((10 ,1))
    return w1, b1, w2, b2

def leaky_relu(z):
    return np.where(z > 0, z, .01 * z)

def leaky_relu_deriv(z):
    return np.where(z > 0, 1, .01)

def softmax(z):
    eps = 1e-10
    return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims=True)

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2) #10, 60000
    return z1, a1, z2, a2

def one_hot(y):
    one_hot_y = np.zeros((np.max(y) + 1, y.size))
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y

def cce(one_hot_y, a):
    eps = 1e-10
    l = - np.sum(one_hot_y * np.log(a + eps)) / one_hot_y.shape[1]
    return l

def accuracy(a, y):
    pred = np.argmax(a, axis = 0, keepdims=True)
    acc = np.sum(y == pred) / y.size * 100
    return acc

def backward(x, one_hot_y, w2, a2, a1, z1):
    dz2 = a2 - one_hot_y
    dw2 = np.dot(dz2, a1.T) / one_hot_y.shape[1]
    db2 = np.sum(dz2, axis = 1, keepdims=True) / one_hot_y.shape[1]
    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1)
    dw1 = np.dot(dz1, x.T) / one_hot_y.shape[1]
    db1 = np.sum(dz1, axis = 1, keepdims=True) / one_hot_y.shape[1]
    return dw1, db1, dw2, db2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha_max, alpha_min, cycle_size, alpha_new, epoch):

    '''

    https://arxiv.org/pdf/1506.01186
    
    Implementing a cyclical `alpha` value
        - hyeprparam: x
        - hyperparam: cycle (cyc)
    
    '''

    cyc = np.floor(1 + epoch / (2 * cycle_size))
    x = np.abs((epoch / cycle_size) - (2 * cyc) + 1)
    alpha = alpha_min + (alpha_max - alpha_min) * np.maximum(0,( 1 - x))


    if epoch >= 430:
        alpha = alpha_new

    # weight update

    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2, alpha

def grad_descent(x, y, w1, b1, w2, b2, epochs, alpha_max, alpha_min, alpha_vec, acc_vec, epoch_vec, cycle_size, alpha_new, file):
    one_hot_y = one_hot(y)
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)

        loss = cce(one_hot_y, a2)
        acc = accuracy(a2, y)

        dw1, db1, dw2, db2 = backward(x, one_hot_y, w2, a2, a1, z1)
        w1, b1, w2, b2, alpha = update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha_max, alpha_min, cycle_size, alpha_new, epoch)
    
        acc_vec.append(acc)
        alpha_vec.append(alpha)
        epoch_vec.append(epoch)

        print(f"Epoch: {epoch}")
        print(f"Loss: {loss}")
        print(f"Acc: {acc}")
        print(f"Alpha: {alpha}\n")

    return w1, b1, w2, b2, alpha_vec, acc_vec

def model(x, y, epochs, alpha_max, alpha_min, alpha_vec, acc_vec, epoch_vec, cycle_size, alpha_new, file):

    try:
        w1, b1, w2, b2 = load_model(file)
        print(f"LOADED MODEL!")

    except FileNotFoundError:
        print(f"MODEL NOT FOUND. INIT NEW MODEL!")
        w1, b1, w2, b2 = init_params()
    
    w1, b1, w2, b2, alpha_vec, acc_vec = grad_descent(x, y, w1, b1, w2, b2, epochs, alpha_max, alpha_min, alpha_vec, acc_vec, epoch_vec, cycle_size, alpha_new, file)

    return w1, b1, w2, b2, alpha_vec, acc_vec

if __name__ == "__main__":
    data = pd.read_csv('../data/fashion-mnist_train.csv')
    data = np.array(data)

    X_train = data[:, 1:785].T / 255
    Y_train = data[:, 0].reshape(1, -1)

    file = 'models/nn.pkl'

    acc_vec = []
    alpha_vec = []
    epoch_vec = []

    w1, b1, w2, b2, alpha_vec, acc_vec = model(X_train, Y_train, epochs = 1000, alpha_max = .55, alpha_min = .08, alpha_vec = alpha_vec, acc_vec = acc_vec, epoch_vec = epoch_vec, cycle_size = 200, alpha_new = .15, file = file)


    plt.figure(figsize=(10, 5))  
    plt.subplot(1, 2, 1)  
    plt.plot(epoch_vec, acc_vec)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Epoch vs Accuracy")

    plt.subplot(1, 2, 2)  
    plt.plot(epoch_vec, alpha_vec)
    plt.xlabel("Epochs")
    plt.ylabel("Alpha")
    plt.title("Epoch vs Alpha")

    plt.tight_layout()  
    plt.show()

