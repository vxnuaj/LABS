'''

https://arxiv.org/pdf/1708.07120

- Neural Network of 64 Params
- batch size of 100, each of 600 samples of the MNIST training set.

The challenge is to increase the training acc to minimum of 

Authors recommend:

- Run the learning rate range test prior to determine the bounds, maximum and minimum, learning rate.
- cycle_size to be 2-10 times larger than the iterations per epoch

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
    
def mini_batch(data, batch_size):
    data_b = np.split(data, batch_size )
    data_b = np.array(data_b)
    return data_b

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

def one_hot(y, batch_size):

    '''
    
    one_hot_y for 3 dimensional ndarray of mnist samples (y) / features. (batches, samples, 1)
    
    '''

    b_one_hot = np.empty((0, 10, int(60000 / batch_size)))
    for i in range(y.shape[0]):
        one_hot_y = np.zeros((np.max(y[i] + 1), y[i].size))
        one_hot_y[y[i], np.arange(y[i].size)] = 1
        b_one_hot = np.concatenate((b_one_hot, one_hot_y[np.newaxis, ...]), axis = 0)
    return b_one_hot

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

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, vdw1, vdb1, vdw2, vdb2, alpha_max, alpha_min, beta_min, beta_max, cycle_size, epoch):

    #cyclical learning rate
    cyc = np.floor(1 + (epoch / (2 * cycle_size)))
    x = np.abs((epoch / cycle_size) - (2 * cyc) + 1)
    alpha = alpha_min + (alpha_max - alpha_min) * (np.maximum(0, (1-x)))

    #cyclical beta
    cyc_b = np.floor(1 + epoch / (2 * cycle_size))
    x = np.abs((epoch / cycle_size) - (2 * cyc_b) + 1)
    beta = beta_min + (beta_max - beta_min) * x

    beta = .99

    #moemntum term
    vdw1 = (beta * vdw1) + (( 1 - beta ) * dw1 * alpha)

    vdb1 = ( beta * vdb1 ) + (( 1 - beta ) * db1 * alpha)

    vdw2 = (beta * vdw2) + (( 1 - beta ) * dw2 * alpha)

    vdb2 = ( beta * vdb2 ) + (( 1 - beta ) * db2 * alpha)


    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2, alpha, vdw1, vdb1, dw2, vdb2, beta

def grad_descent(x, y, w1, b1, w2, b2, epochs, alpha_max, alpha_min, beta_max, beta_min, cycle_size, batch_size, file):
    one_hot_y = one_hot(Y_train, batch_size = batch_size) #10, 10, 6000

    alpha_vec = []
    beta_vec = []
    epoch_vec = []
    loss_vec = []
    acc_vec = []
    iteration_vec = []
    iteration = 0

    vdw1, vdb1, vdw2, vdb2 = 0, 0, 0, 0
    
    for epoch in range(epochs):
        for i in range(x.shape[0]):
            iteration += 1
            z1, a1, z2, a2 = forward(x[i], w1, b1, w2, b2)

            loss = cce(one_hot_y[i], a2)
            acc = accuracy(a2, y[i])

            dw1, db1, dw2, db2 = backward(x[i], one_hot_y[i], w2, a2, a1, z1)
            w1, b1, w2, b2, alpha, vdw1, vdb1, dw2, vdb2, beta = update(w1, b1, w2, b2, dw1, db1, dw2, db2, vdw1, vdb1, vdw2, vdb2, alpha_max, alpha_min, beta_max, beta_min, cycle_size, epoch)
        
            if i % 20 == 0:
                print(f"Epoch: {epoch} | Iteration: {i}")
                print(f"Loss: {loss}")
                print(f"Acc: {acc}")
                print(f"Alpha: {alpha}")
                print(f"Beta: {beta}\n")

            alpha_vec.append(alpha)
            beta_vec.append(beta)
            epoch_vec.append(epoch)
            iteration_vec.append(iteration)
            loss_vec.append(loss)
            acc_vec.append(acc)

    return w1, b1, w2, b2, alpha_vec, beta_vec, epoch_vec, iteration_vec, loss_vec, acc_vec

def model(x, y, epochs, alpha_max, alpha_min, beta_max, beta_min, cycle_size, batch_size, file):
    
    try:
        w1, b1, w2, b2 = load_model(file)
        print(f"LOADED MODEL!")

    except FileNotFoundError:
        print(f"MODEL NOT FOUND. INIT NEW MODEL!")
        w1, b1, w2, b2 = init_params()
    
    w1, b1, w2, b2, alpha_vec, beta_vec, epoch_vec, iteration_vec, loss_vec, acc_vec = grad_descent(x, y, w1, b1, w2, b2, epochs, alpha_max, alpha_min, beta_max, beta_min, cycle_size, batch_size, file)

    return w1, b1, w2, b2, alpha_vec, beta_vec, epoch_vec, iteration_vec, loss_vec, acc_vec

if __name__ == "__main__":
    data = pd.read_csv('../data/fashion-mnist_train.csv')
    data = np.array(data)

    batch_size = 100

    data_b = mini_batch(data , batch_size=batch_size)
    X_train = data_b[:, :, 1:786]
    X_train = X_train / 255
    X_train = X_train.reshape(batch_size, 784, int(60000 / batch_size)) 
    
    Y_train = data_b[:, :, 0].reshape(batch_size, 1, int(60000 / batch_size)) 

    file = 'models/nn.pkl'

    print(X_train.shape)
    print(Y_train.shape)

    w1, b1, w2, b2, alpha_vec, beta_vec, epoch_vec, iteration_vec, loss_vec, acc_vec = model(X_train, Y_train, epochs = 90, alpha_max = .6, alpha_min = .1, beta_max = .99, beta_min = .9, cycle_size = 10, batch_size=batch_size, file = file)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)


    axs[0].plot(iteration_vec, acc_vec, label='Accuracy')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    axs[1].plot(iteration_vec, alpha_vec, label='Alpha')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Learning Rate')
    axs[1].legend()

    axs[2].plot(iteration_vec, beta_vec, label='Alpha')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Momentum Term')
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()