'''

Implementing BatchNorm on mini-batch gradient descent (mini-batches of 6k samples, FashionMNIST), but this time with RMSProp.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

def save_model(file, w1, g1, b1, w2, g2, b2, a1, a2):
    with open(file, 'wb') as f:
        pkl.dump(( w1, g1, b1, w2, g2, b2, a1, a2), f)

def load_model(file):
    with open(file, 'rb') as f:
        w1, g1, b1, w2, g2, b2, a1, a2 = pkl.load(f)
        return w1, g1, b1, w2, g2, b2, a1, a2

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
    g1 = np.ones((32, 1))
    b1 = np.zeros((32, 1))
    w2 = rng.normal(size = (10 ,32)) * np.sqrt(1/ 784)
    g2 = np.ones((10, 1))
    b2 = np.zeros((10 ,1))
    return w1, g1, b1, w2, g2, b2

def leaky_relu(z):
    return np.where(z>0, z, .01 * z)

def leaky_relu_deriv(z):
    return np.where(z>0, 1, .01)

def softmax(z):
    eps = 1e-8
    return np.exp(z + eps) / (np.sum(np.exp(z + eps), axis = 0, keepdims=True))

def batch_norm(z):
    eps = 1e-8
    mu = np.mean(z, axis = 1, keepdims=True)
    var = np.var(z, axis = 1, keepdims=True)
    z = (z - mu) / np.sqrt(var + eps)
    return z, np.sqrt(var + eps)

def forward(x, w1, g1, b1, w2, g2, b2):
    z1 = np.dot(w1, x)
    b_z1_norm, std1 = batch_norm(z1)
    z1_norm = (g1 * b_z1_norm) + b1
    a1 = leaky_relu(z1_norm) 
    z2 = np.dot(w2, a1)
    b_z2_norm, std2 = batch_norm(z2)
    z2_norm = (g2 * b_z2_norm) + b2
    a2 = softmax(z2_norm)
    return z1, z1_norm, b_z1_norm, a1, z2, z2_norm, b_z2_norm, a2, std1, std2

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

def backward(x, one_hot, a2, a1, w2, b_z2_norm, b_z1_norm, z1_norm, g2, g1, std2, std1):
    eps = 1e-8

    ''' Layer 2 Backprop '''

    dz2_norm = a2 - one_hot #10, (60000 / batch_size)
    dg2 = dz2_norm * b_z2_norm  #10, (60000 / batch_size)
    db2 = dz2_norm  #10, (60000 / batch_size)
    dz2 = dz2_norm * g2 * (1 / np.abs(std2 + eps)) #10, (60000 / batch_size)
    dw2 = np.dot(dz2, a1.T) / x.shape[1] #  #(10, (60000 / batch_size) • (60000 / batch_size, 32)) -> 10, 32

    ''' Layer 1 Backprop'''

    dz1_norm = np.dot(w2.T, dz2) * leaky_relu_deriv(z1_norm) # ( (32, 10) • (10, 60000 / batch_size)) -> 32, 6000
    dg1 = dz1_norm * b_z1_norm  # 32, (60000 / batch_size)
    db1 = dz1_norm # 32, (60000 / batch_size)
    dz1 = dz1_norm * g1 * (1 / np.abs(std1 + eps)) # 32, (60000 / batch_size)
    dw1 = np.dot(dz1, x.T) / x.shape[1] # ( (32, (60000 / batch_size)) • (60000 / batch_size, 784)) -> 32, 784

    return dw2, dg2, db2, dw1, dg1, db1

def update(w2, g2, b2, w1, g1, b1, dw2, dg2, db2, dw1, dg1, db1, sdw1, sdg1, sdb1, sdw2, sdg2, sdb2, vdw1, vdg1, vdb1, vdw2, vdg2, vdb2, alpha, beta1, beta2):

    eps = 1e-8

    '''
    
    Computing EWAs of the gradients

    '''

    vdw1 = (beta1 * vdw1) + (1 - beta1) * dw1
    vdg1 = (beta1 * vdg1) + (1 - beta1) * dg1
    vdb1 = (beta1 * vdb1) + (1 - beta1) * db1

    vdw2 = (beta1 * vdw2) + (1 - beta1) * dw2
    vdg2 = (beta1 * vdg2) + (1 - beta1) * dg2
    vdb2 = (beta1 * vdb2) + (1 - beta1) * db2

    '''
    
    Computing EWAs based on squared gradients.
    
    '''

    sdw1 = (beta2 * sdw1) + (1 - beta2) * np.square(dw1)
    sdg1 = (beta2 * sdg1) + (1 - beta2) * np.square(dg1)
    sdb1 = (beta2 * sdb1) + (1 - beta2) * np.square(db1)
    
    sdw2 = (beta2 * sdw2) + (1 - beta2) * np.square(dw2)
    sdg2 = (beta2 * sdg2) + (1 - beta2) * np.square(dg2)
    sdb2 = (beta2 * sdb2) + (1 - beta2) * np.square(db2)
    
    
    '''
    
    Adam's Update rule
    
    '''

    w2 = w2 - (alpha / (np.sqrt(sdw2 + eps))) * vdw2
    g2 = g2 - (alpha / (np.sqrt(sdg2 + eps))) * vdg2
    b2 = b2 -  (alpha / (np.sqrt(sdb2 + eps))) * vdb2
    w1 = w1 -  (alpha / (np.sqrt(sdw1 + eps))) * vdw1
    g1 = g1 -  (alpha / (np.sqrt(sdg1 + eps))) * vdg1
    b1 = b1 -  (alpha / (np.sqrt(sdb1 + eps))) * vdb1
    return w2, g2, b2, w1, g1, b1, sdw1, sdg1, sdb1, sdw2, sdg2, sdb2, vdw1, vdg1, vdb1, vdw2, vdg2, vdb2,

def gradient_descent(x, y, w1, g1, b1, w2, g2, b2, epochs, alpha, beta1, beta2, batch_num, file):
    one_hot_y_b = one_hot(y, batch_num)

    acc_vec = []
    loss_vec = []
    epoch_vec = []

    sdw1, sdg1, sdb1, sdw2, sdg2, sdb2 = 0, 0, 0, 0, 0, 0 # initializing EWAs of gradietns ** 2 to 0, given no history
    vdw1, vdg1, vdb1, vdw2, vdg2, vdb2 = 0, 0, 0, 0, 0, 0 # initializing EWAs to 0, given no history


    for epoch in range(epochs):
        for i in range(x.shape[0]):
            z1, z1_norm, b_z1_norm, a1, z2, z2_norm, b_z2_norm, a2, std1, std2 = forward(x[i], w1, g1, b1, w2, g2, b2)

            loss = CCE(one_hot_y_b[i], a2)
            accuracy = acc(y[i], a2)

            dw2, dg2, db2, dw1, dg1, db1 = backward(x[i], one_hot_y_b[i], a2, a1, w2, b_z2_norm, b_z1_norm, z1_norm, g2, g1, std2, std1)
            w2, g2, b2, w1, g1, b1, sdw1, sdg1, sdb1, sdw2, sdg2, sdb2, vdw1, vdg1, vdb1, vdw2, vdg2, vdb2, = update(w2, g2, b2, w1, g1, b1, dw2, dg2, db2, dw1, dg1, db1, sdw1, sdg1, sdb1, sdw2, sdg2, sdb2, vdw1, vdg1, vdb1, vdw2, vdg2, vdb2, alpha, beta1, beta2)
            
            acc_vec.append(accuracy)
            loss_vec.append(loss)
            epoch_vec.append(epoch)

            if i % 2 == 0:
                print(f"Epoch: {epoch} | Iteration: {i}")
                print(f"Accuracy: {accuracy}")
                print(f"Loss: {loss}")
                print(f"z2: {np.mean(z2)}\n") #exploding?
    
    save_model(file, w1, g1, b1, w2, g2, b2, a1, a2)

    return w1, g1, b1, w2, g2, b2, acc_vec, loss_vec, epoch_vec


def model(x, y, epochs, alpha, beta1, beta2, batch_num, file):
    try:
        w1, g1, b1, w2, g2, b2 = load_model(file)
        print("Loaded model!")
    except FileNotFoundError:
        print("Model not found initializing new params!")
        w1, g1, b1, w2, g2, b2 = init_params()
    w1, g1, b1, w2, g2, b2, acc_vec, loss_vec, epoch_vec = gradient_descent(x, y, w1, g1, b1, w2, g2, b2, epochs, alpha, beta1, beta2, batch_num, file)
    return w1, g1, b1, w2, g2, b2, acc_vec, loss_vec, epoch_vec
    

if __name__ == "__main__":

    batch_num = 10
    epochs = 150
    alpha = .05
    beta1 = .9
    beta2 = .99
    file = 'models/viz1l.pkl'
    data = pd.read_csv("../data/fashion-mnist_train.csv")

    #X_train, Y_train = minibatches_process(data, batch_num)

    #w1, g1, b1, w2, g2, b2, acc_vec, loss_vec, epoch_vec = model(X_train, Y_train, epochs, alpha, beta1, beta2, batch_num, file)



    w1, g1, b1, w2, g2, b2, a1, a2 = load_model(file)

    print(w1.shape)

    i = 31

    weights = w1[i].reshape(28, 28)

    weights = pd.DataFrame(weights)

    weights.to_csv(f"IndivWeights/L1W{i}.csv")

    plt.imshow(weights, cmap = 'seismic')
    plt.show()

    '''
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)


    axs[0].plot(epoch_vec, loss_vec, label='Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(epoch_vec, acc_vec, label='Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()
    '''
