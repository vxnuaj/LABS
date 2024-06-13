'''

Implementing BatchNorm on mini-batch gradient descent (mini-batches of 10k samples, FashionMNIST)

'''


import numpy as np
import pandas as pd
import pickle

def mini_batches(data, batch_size):

    #should return data in dims (batches, features, samples)    
    # takes in data with shape (samples, features)

    data_batched = np.split(data, indices_or_sections=batch_size)
    data_batched = np.array(data_batched)
    data_batched = data_batched.reshape(batch_size,data.shape[1],  -1)
    return data_batched

def save_model(file, w1, b1, w2, b2):
    with open(file, 'wb') as f:
        pickle.dump((w1, b1, w2, b2), f)

def load_model(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def init_params():
    rng = np.random.default_rng(seed = 1)
    w1 =rng.normal(size = (64, 784)) * np.sqrt(1/784)
    gamma1 = np.ones((64, 1))
    beta1 = np.zeros((64, 1))

    w2 = rng.normal(size = (10, 64)) * np.sqrt(1/784)
    gamma2 = np.ones((10, 1))
    beta2 = np.zeros((10, 1))
    return w1, w2, gamma1, beta1, gamma2, beta2

def leaky_relu(z):
    return np.where(z > 0, z, .01 * z)

def leaky_relu_deriv(z):
    return np.where(z > 0, 1, .01)

def softmax(z):
    eps = 1e-10
    return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims=True)

def batchnorm(z, gamma, beta):
    eps = 1e-8
    mu = np.sum(z, axis = 0, keepdims=True) / z.shape[0]
    std_2 = np.sum((z - mu) ** 2, axis = 0, keepdims=True) / z.shape[0]
    z_norm = (z - mu) / (np.sqrt(std_2 + eps))
    z_norm_a = gamma * z_norm + beta
    return z_norm_a, z_norm

def forward(x, w1, w2, gamma1, gamma2, beta1, beta2):
    z1 = np.dot(w1, x)
    z1, z1_norm_a = batchnorm(z1, gamma1, beta1)
    a1 = leaky_relu(z1)
    z2 = np.dot(w2, a1)
    z2, z2_norm_a = batchnorm(z2, gamma2, beta2)
    a2 = softmax(z2) #10, 60000
    return z1, a1, z2, a2, z1_norm_a, z2_norm_a

def one_hot(y):
    b_one_hot = np.empty((0, 10, 600))
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

def backward(x, one_hot_y, w2, a2, a1, z1, z1_norm_a, z2_norm_a):
    dz2 = a2 - one_hot_y
    dw2 = np.dot(dz2, a1.T) / one_hot_y.shape[1]
    dgamma2 = np.sum(dz2 * z2_norm_a, axis = 1, keepdims = True) / one_hot_y.shape[1]
    dbeta2 = np.sum(dz2, axis = 1, keepdims = True) / one_hot_y.shape[1]

    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1_norm_a)
    dw1 = np.dot(dz1, x.T) / one_hot_y.shape[1]
    dgamma1 = np.sum(dz1 * z1_norm_a, axis = 1, keepdims=True) / one_hot_y.shape[1]
    dbeta1 = np.sum(dz1, axis = 1, keepdims=True) / one_hot_y.shape[1]

    return dw1, dw2, dgamma1, dgamma2, dbeta1, dbeta2

def update(w1, w2,gamma1, gamma2, beta1, beta2, dw1, dw2, dgamma1, dgamma2, dbeta1, dbeta2, alpha):
    w1 = w1 - alpha * dw1
    gamma1 = gamma1 - alpha * dgamma1
    beta1 = beta1 - alpha * dbeta1
    
    w2 = w2 - alpha * dw2
    gamma2 = gamma2 - alpha * dgamma2
    beta2 = beta2 - alpha * dbeta2
    return w1, w2, gamma1, gamma2, beta1, beta2

def grad_descent(x, y, w1, w2, gamma1, beta1, gamma2, beta2, epochs, alpha, file):
    one_hot_y = one_hot(y)
    for epoch in range(epochs):
        for i in range(x.shape[0]):
            z1, a1, z2, a2, z1_norm_a, z2_norm_a = forward(x[i], w1, w2, gamma1, gamma2, beta1, beta2)

            loss = cce(one_hot_y[i], a2)
            acc = accuracy(a2, y[i])

            dw1, dw2, dgamma1, dgamma2, dbeta1, dbeta2 = backward(x[i], one_hot_y[i], w2, a2, a1, z1, z1_norm_a, z2_norm_a)
            w1, w2, gamma1, gamma2, beta1, beta2 = update(w1, w2,gamma1, gamma2, beta1, beta2, dw1, dw2, dgamma1, dgamma2, dbeta1, dbeta2, alpha)
        
            print(f"Iteration: {i}")
            print(f"Epoch: {epoch}")
            print(f"Loss: {loss}")
            print(f"Acc: {acc}")
            print(f"Beta1: {np.max(beta1)}")
            print(f"Beta2: {np.max(beta2)}\n")

    return w1, w2, gamma1, gamma2, beta1, beta2,

def model(x, y, epochs, alpha, file):

    try:
        w1, w2, gamma1, gamma2, beta1, beta2 = load_model(file)
        print(f"LOADED MODEL!")

    except FileNotFoundError:
        print(f"MODEL NOT FOUND. INIT NEW MODEL!")
        w1, w2, gamma1, beta1, gamma2, beta2 = init_params()
    
    w1, w2, gamma1, gamma2, beta1, beta2 = grad_descent(x, y, w1, w2, gamma1, beta1, gamma2, beta2, epochs, alpha, file)

    return w1, w2, gamma1, gamma2, beta1, beta2

if __name__ == "__main__":
    data = pd.read_csv('data/fashion-mnist_train.csv')
    data = np.array(data)

    data_batched = mini_batches(data, batch_size = 10) # 10, 785, 6000

    X_train = data_batched[:, 1:786, :] / 255 # 10, 784, 6000
    Y_train = data_batched[:, 0, :].reshape(10, -1, 6000) # 10, 1, 6000

    file = 'models/nn.pkl'
    epochs = 100
    alpha = .1

    w1, w2, gamma1, gamma2, beta1, beta2 = model(X_train, Y_train, epochs, alpha, file)
