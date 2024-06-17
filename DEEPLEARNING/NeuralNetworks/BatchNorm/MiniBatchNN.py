'''

MiniBatched NN, batch size 6k, batch_num: 10

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def minibatches_process(data, batch_num):

    ''' Returns processed and batched data (for MNIST), split into Labels (Y) and Features (X)'''

    data = np.array(data).T
    x = data[1:786, :] # 784, 60000
    y = data[0, :].reshape(1, 60000) # 1, 60000

    x_batched = np.array(np.split(x, batch_num, axis = 1)) / 255 # 10, 784, (60000 / batch_num)
    y_batched = np.array(np.split(y,  batch_num, axis = 1)) # 10, 1, (60000 / batch_num)
    return x_batched, y_batched

def init_params():
    rng = np.random.default_rng(seed = 1)
    w1 = rng.normal(size = (32, 784)) * np.sqrt(1/784)
    b1 = np.zeros((32, 1))
    w2 = rng.normal(size = (10 ,32)) * np.sqrt(1/ 784)
    b2 = np.zeros((10 ,1))
    return w1, b1, w2, b2

def leaky_relu(z):
    return np.where(z>0, z, .01 * z)

def leaky_relu_deriv(z):
    return np.where(z>0, 1, .01)

def softmax(z):
    eps = 1e-8
    return np.exp(z + eps) / (np.sum(np.exp(z + eps), axis = 0, keepdims=True))

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(y, batch_num):
    one_hot_y_b = np.empty(shape = (0, 10, int(60000 / batch_num)))
    for i in range(y.shape[0]):
        one_hot_y = np.zeros((np.max(y[i] + 1), y[i].size)) # (10, (60000 / batch_num))
        one_hot_y[y[i], np.arange(y[i].size)] = 1
        one_hot_y_b = np.concatenate((one_hot_y_b, one_hot_y[np.newaxis, ...]), axis = 0)
    return one_hot_y_b # 10, 10, (600000 / batch_num)

def CCE(one_hot, a):
    eps = 1e-8
    loss = - np.sum(one_hot * np.log(a + eps)) / one_hot.shape[1]
    return loss

def acc(y, a2):
    pred = np.argmax(a2, axis = 0)
    accuracy = np.sum(y == pred) / y.size * 100
    return accuracy

def backward(x, one_hot, a2, a1, w2, z1):
    eps = 1e-8

    ''' Layer 2 Backprop '''

    dz2 = a2 - one_hot
    dw2 = np.dot(dz2, a1.T) / x.shape[1]
    db2 = np.sum(dz2, axis = 1, keepdims=True) /  x.shape[1]
    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1)
    dw1 = np.dot(dz1, x.T) /  x.shape[1]
    db1 = np.sum(dz1, axis=1, keepdims=True) /  x.shape[1]
    return dw2, db2, dw1, db1

def update(w2, b2, w1, b1, dw2, db2, dw1, db1, alpha):
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    return w2, b2, w1, b1

def gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha, batch_num):
    one_hot_y_b = one_hot(y, batch_num)

    acc_vec = []
    loss_vec = []
    epoch_vec = []
    w1_vec = []
    dw1_vec = []

    for epoch in range(epochs):
        for i in range(x.shape[0]):
            z1, a1, z2, a2 = forward(x[i], w1, b1, w2, b2)

            loss = CCE(one_hot_y_b[i], a2)
            accuracy = acc(y[i], a2)

            dw2, db2, dw1, db1 = backward(x[i], one_hot_y_b[i], a2, a1, w2, z1)
            w2, b2, w1, b1 = update(w2, b2, w1, b1, dw2, db2, dw1, db1, alpha)
            
            acc_vec.append(accuracy)
            loss_vec.append(loss)
            epoch_vec.append(epoch)
            w1_vec.append(np.mean(w1))
            dw1_vec.append(np.mean(dw1))

            if i % 2 == 0:
                print(f"Epoch: {epoch} | Iteration: {i}")
                print(f"Accuracy: {accuracy}")
                print(f"Loss: {loss}\n")

    return w1, b1, w2, b2, acc_vec, loss_vec, epoch_vec, w1_vec, dw1_vec


def model(x, y, epochs, alpha, batch_num):
    w1, b1, w2, b2= init_params()
    w1, b1, w2, b2, acc_vec, loss_vec, epoch_vec, w1_vec, dw1_vec = gradient_descent(x, y, w1, b1, w2, b2, epochs, alpha, batch_num)
    return w1, b1, w2, b2, acc_vec, loss_vec, epoch_vec, w1_vec, dw1_vec
    

if __name__ == "__main__":

    batch_num = 10
    epochs = 100
    alpha = .1
    data = pd.read_csv("../data/fashion-mnist_train.csv")

    X_train, Y_train = minibatches_process(data, batch_num)

    w1, b1, w2, b2, acc_vec, loss_vec, epoch_vec, w1_vec, dw1_vec = model(X_train, Y_train, epochs, alpha, batch_num)

    w1 = np.array(w1_vec)
    dw1 = np.array(dw1_vec)
    print("w1 avg", np.mean(w1))
    print("dw1 avg", np.mean(dw1))

    '''fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)'''


    '''axs[0].plot(epoch_vec, dw2_vec, label='dw2')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('dw2')
    axs[0].legend()

    axs[1].plot(epoch_vec, w2_vec, label='w2')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('w2')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()'''

