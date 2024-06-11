'''

Implementing Cyclical Learning Rate with a Cyclical Momentum Term.

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
    w1 = rng.normal(size = (32, 784)) * np.sqrt(1/784)
    b1 = np.zeros((32, 1))
    w2 = rng.normal(size = (10, 32)) * np.sqrt(1/784)
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

def backward(x, one_hot_y, w2, a2, a1, z1, vdw1, vdb1, vdw2, vdb2, alpha, beta_max, beta_min, cycle_size, epoch):
    dz2 = a2 - one_hot_y
    dw2 = np.dot(dz2, a1.T) / one_hot_y.shape[1]
    db2 = np.sum(dz2, axis = 1, keepdims=True) / one_hot_y.shape[1]
    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1)
    dw1 = np.dot(dz1, x.T) / one_hot_y.shape[1]
    db1 = np.sum(dz1, axis = 1, keepdims=True) / one_hot_y.shape[1]

    '''
    
    Cyclical Momentum
    
    '''

    cyc_b = np.floor(1 + epoch / (2 * cycle_size))
    x = np.abs((epoch / cycle_size) - (2 * cyc_b) + 1)
    beta = beta_min + (beta_max - beta_min) * x

    '''
    
    Updating the momentum term
    
    '''

    vdw1 = (beta * vdw1) + (( 1 - beta ) * dw1 * alpha)

    vdb1 = ( beta * vdb1 ) + (( 1 - beta ) * db1 * alpha)

    vdw2 = (beta * vdw2) + (( 1 - beta ) * dw2 * alpha)

    vdb2 = ( beta * vdb2 ) + (( 1 - beta ) * db2 * alpha)


    return dw1, db1, dw2, db2, vdw1, vdb1, vdw2, vdb2, beta

def update(w1, b1, w2, b2, vdw1, vdb1, vdw2, vdb2, alpha_max, alpha_min, cycle_size, epoch):

    '''
    
    Cyclical Learning Rate

    '''

    cyc = np.floor(1 + (epoch / (2 * cycle_size)))
    x = np.abs((epoch / cycle_size) - (2 * cyc) + 1 )
    alpha = alpha_min + (alpha_max - alpha_min) * np.maximum(0, (1 - x))

    if epoch > 550 and alpha > .55:
        alpha = .55

    w1 = w1 - alpha * vdw1
    b1 = b1 - alpha * vdb1
    w2 = w2 - alpha * vdw2
    b2 = b2 - alpha * vdb2
    return w1, b1, w2, b2, alpha

def grad_descent(x, y, w1, b1, w2, b2, epochs, alpha_max, alpha_min, cycle_size, beta_max, beta_min, file):
    one_hot_y = one_hot(y)

    #init the velocity terms to 0

    vdw1, vdb1, vdw2, vdb2 = 0, 0, 0, 0

    #init alpha to .08 for first `backward` pass

    alpha = .08

    #lists to plot:

    acc_vals = []
    loss_vals = []
    alpha_vals = []
    beta_vals = []
    epochs_vals = []

    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)

        loss = cce(one_hot_y, a2)
        acc = accuracy(a2, y)

        dw1, db1, dw2, db2, vdw1, vdb1, vdw2, vdb2, beta = backward(x, one_hot_y, w2, a2, a1, z1, vdw1, vdb1, vdw2, vdb2, alpha, beta_max, beta_min, cycle_size, epoch)
        w1, b1, w2, b2, alpha = update(w1, b1, w2, b2, vdw1, vdb1, vdw2, vdb2, alpha_max, alpha_min, cycle_size, epoch)
    
        print(f"Epoch: {epoch}")
        print(f"Loss: {loss}")
        print(f"Acc: {acc}")
        print(f"Alpha: {alpha}")
        print(f"Beta: {beta}\n")

        acc_vals.append(acc)
        loss_vals.append(loss)
        alpha_vals.append(alpha)
        beta_vals.append(beta)
        epochs_vals.append(epoch)

    return w1, b1, w2, b2, acc_vals, loss_vals, alpha_vals, beta_vals, epochs_vals

def model(x, y, epochs, alpha_max, alpha_min, cycle_size, beta_max, beta_min, file):

    try:
        w1, b1, w2, b2 = load_model(file)
        print(f"LOADED MODEL!")

    except FileNotFoundError:
        print(f"MODEL NOT FOUND. INIT NEW MODEL!")
        w1, b1, w2, b2 = init_params()
    
    w1, b1, w2, b2, acc_vals, loss_vals, alpha_vals, beta_vals, epochs_vals = grad_descent(x, y, w1, b1, w2, b2, epochs, alpha_max, alpha_min, cycle_size, beta_max, beta_min, file)

    return w1, b1, w2, b2, acc_vals, loss_vals, alpha_vals, beta_vals, epochs_vals

if __name__ == "__main__":
    data = pd.read_csv('../data/fashion-mnist_train.csv')
    data = np.array(data)

    X_train = data[:, 1:785].T / 255
    Y_train = data[:, 0].reshape(1, -1)

    file = 'models/nn.pkl'

    w1, b1, w2, b2, acc_vals, loss_vals, alpha_vals, beta_vals, epochs_vals = model(X_train, Y_train, epochs = 1000, alpha_max = .68, alpha_min = .15, cycle_size = 200, beta_max = .99, beta_min = .75, file = file)

    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axs[0].plot(epochs_vals, acc_vals, label='Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    axs[1].plot(epochs_vals, loss_vals, label='Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    axs[2].plot(epochs_vals, alpha_vals, label='Alpha')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Alpha')
    axs[2].legend()

    axs[3].plot(epochs_vals, beta_vals, label='Beta')
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('Beta')
    axs[3].legend()

    plt.tight_layout()
    plt.show()