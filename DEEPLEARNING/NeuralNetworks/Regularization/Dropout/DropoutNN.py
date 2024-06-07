''''
Implementing Dropout
- Dropping 50% in 1st Hidden Layer
- Dropping 0% in the output layer

'''

import numpy as np
import pandas as pd
import pickle

def save_model(file, w1, b1, w2, b2):
    with open(file, 'wb') as f:
        pickle.dump((w1, b1, w2, b2), f)

def load_model(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def init_params():
    w1 = np.random.randn(64, 784) * np.sqrt(1/784)
    b1 = np.zeros((64, 1))
    w2 = np.random.randn(10, 64) * np.sqrt(1/784)
    b2 = np.zeros((10 ,1))
    return w1, b1, w2, b2

def leaky_relu(z):
    return np.where(z > 0, z, .01 * z)

def leaky_relu_deriv(z):
    return np.where(z > 0, 1, .01)

def softmax(z):
    eps = 1e-10
    return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims=True)

def forward(x, w1, b1, w2, b2, p):
    z1 = np.dot(w1, x) + b1
    a1 = leaky_relu(z1) # 64, 60000
    d1 = np.random.rand(a1.shape[0], a1.shape[1])
    d1 = d1 > p
    '''
    Using `np.random.rand()` as we want values between 0 and 1, not between -1 and 1.
    
    We want to do so in order to be able to uniformly distribute values between 0 and 1, allowing
    us to effectively use probability `p` to drop a set of weights. 
    
    Otherwise, if generated using a normal distribution (np.random.randn) probability `p` 
    wouldn't be able to reliably get a set of neurons to be equal to 0 as the values won't be 
    distributed evenly (uniformly). The actual % of neurons dropped to 0 will be non-fidel to probability `p`.
    '''
    a1 = np.multiply(a1, d1)
    a1 = a1 / p
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2) #10, 60000
    return z1, a1, z2, a2

def predict(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
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
    db2 = np.sum(dz2, axis = 1, keepdims = True) / one_hot_y.shape[1]
    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1)
    dw1 = np.dot(dz1, x.T) / one_hot_y.shape[1]
    db1 = np.sum(dz1, axis = 1, keepdims=True) / one_hot_y.shape[1]
    return dw1, db1, dw2, db2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def grad_descent(x, y, w1, b1, w2, b2, epochs, alpha, p, file):
    one_hot_y = one_hot(y)
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(x, w1, b1, w2, b2, p)

        loss = cce(one_hot_y, a2)
        acc = accuracy(a2, y)

        dw1, db1, dw2, db2 = backward(x, one_hot_y, w2, a2, a1, z1)
        w1, b1, w2, b2 = update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
    
        print(f"Epoch: {epoch}")
        print(f"Loss: {loss}")
        print(f"Acc: {acc}\n")

    save_model(file, w1, b1, w2, b2)
    print(f"Saved model!")

    return w1, b1, w2, b2

def model(x, y, epochs, alpha, p, file):

    try:
        w1, b1, w2, b2 = load_model(file)
        print(f"LOADED MODEL!")

    except FileNotFoundError:
        print(f"MODEL NOT FOUND. INIT NEW MODEL!")
        w1, b1, w2, b2 = init_params()
    
    w1, b1, w2, b2 = grad_descent(x, y, w1, b1, w2, b2, epochs, alpha, p, file)

    return w1, b1, w2, b2

if __name__ == "__main__":
    data = pd.read_csv('data/fashion-mnist_train.csv')
    data = np.array(data)

    X_train = data[:, 1:785].T / 255
    Y_train = data[:, 0].reshape(1, -1)

    file = 'models/dropoutNN.pkl'

    w1, b1, w2, b2 = model(X_train, Y_train, 500, .1, .5, file)

