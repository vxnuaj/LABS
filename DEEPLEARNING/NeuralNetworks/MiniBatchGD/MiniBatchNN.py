import numpy as np
import pandas as pd
import pickle

def save_model(file, w1, b1, w2, b2):
    with open(file, 'wb') as f:
        pickle.dump((w1, b1, w2, b2), f)

def load_model(file):
    try:
        with open(file, 'rb') as f:
            print("Loaded model!")
            return pickle.load(f)
    except FileNotFoundError:
        print("Model not found! Initializing new model!")
        w1, b1, w2, b2 = init_params()
        return w1, b1, w2, b2    

def mini_batches(data, n):

    ydata = data[0, :].reshape(1, -1)
    xdata = data[1:, :]

    xbatched = np.array(np.split(xdata, n, axis = 1)) / 255
    ybatched = np.array(np.split(ydata, n, axis = 1))

    return xbatched, ybatched

def init_params():
    w1 = np.random.randn(32, 784) * np.sqrt(1/784)
    b1 = np.zeros((32, 1))
    w2 = np.random.randn(10, 32) * np.sqrt(1/ 784)
    b2 = np.zeros((10 ,1))
    return w1, b1, w2, b2

def leaky_relu(z):
    return np.where(z > 0, z, z * .01)

def relu_deriv(z):
    return np.where(z > 0, 1, .01)

def softmax(z):
    eps = 1e-10
    return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims = True)

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(y):
    b_one_hot = np.empty((0, 10, 10000))
    for i in range(y.shape[0]):
        one_hot_y = np.zeros((np.max(y[i] + 1), y[i].size))
        one_hot_y[y[i], np.arange(y[i].size)] = 1
        b_one_hot = np.concatenate((b_one_hot, one_hot_y[np.newaxis, ...]), axis = 0)
    return b_one_hot

def cce(one_hot_y, a):
    eps = 1e-10
    loss = - np.sum(one_hot_y * np.log(a + eps)) / one_hot_y.shape[1]
    return loss

def acc(y, a2):
    pred = np.argmax(a2, axis = 0)
    accuracy = np.sum(y == pred) / y.size * 100
    return accuracy

def backward(x, one_hot_y, a2, a1, w2, z1):
    dz2 = a2 - one_hot_y
    dw2 = np.dot(dz2, a1.T) / one_hot_y.shape[1]
    db2 = np.sum(dz2, axis = 1 , keepdims=True) / one_hot_y.shape[1]
    dz1 = np.dot(w2.T, dz2) * relu_deriv(z1)
    dw1 = np.dot(dz1, x.T) / one_hot_y.shape[1]
    db1 = np.sum(dz1, axis = 1, keepdims=True) / one_hot_y.shape[1]
    return dw1, db1, dw2, db2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def minibatch_descent(x, y, w1, b1, w2, b2, epochs, alpha, file):
    b_one_hot = one_hot(y)
    for epoch in range(epochs):
        for i in range(x.shape[0]):
            z1, a1, z2, a2 = forward(x[i], w1, b1, w2, b2)

            loss = cce(b_one_hot[i], a2)
            accuracy = acc(y[i], a2)

            dw1, db1, dw2, db2 = backward(x[i], b_one_hot[i], a2, a1, w2, z1)
            w1, b1, w2, b2 = update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

            print(f"Epoch: {epoch}")
            print(f"Loss: {loss}")
            print(f"Accuracy: {accuracy}\n")

    save_model(file, w1, b1, w2, b2)
    return w1, b1, w2, b2


def model(x, y, alpha, epochs, file):
    w1, b1, w2, b2 = load_model(file)
    w1, b1, w2, b2 = minibatch_descent(x, y, w1, b1, w2, b2, epochs, alpha, file)    
    return w1, b1, w2, b2


if __name__ == "__main__":
    
    data = pd.read_csv('data/fashion-mnist_train.csv')
    data = np.array(data).T #785, 60000

    xbatched, ybatched = mini_batches(data , int(6))

    w1, b1, w2, b2 = model(xbatched, ybatched, .1, 1000, 'models/MiniBatchNN.pkl')