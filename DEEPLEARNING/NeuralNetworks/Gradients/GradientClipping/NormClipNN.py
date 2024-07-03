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

def backward(x, one_hot_y, w2, a2, a1, z1, k):
    dz2 = a2 - one_hot_y
    dw2 = np.dot(dz2, a1.T) / one_hot_y.shape[1]
    db2 = np.sum(dz2, axis = 1, keepdims=True) / one_hot_y.shape[1]
    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1)
    dw1 = np.dot(dz1, x.T) / one_hot_y.shape[1]
    db1 = np.sum(dz1, axis = 1, keepdims=True) / one_hot_y.shape[1]
   
    # implementation of gradient norm clipping
   
    print(f"Orig DW2: {np.max(dw2), np.min(dw2)}")
     
    dw2_norm = np.linalg.norm(dw2, ord = 2) 
    db2_norm = np.linalg.norm(db2, ord = 2) 
    dw1_norm = np.linalg.norm(dw1, ord = 2) 
    db1_norm = np.linalg.norm(db1, ord = 2)
  
    dw2 = np.where(dw2_norm > k, (dw2 / dw2_norm) * k, dw2)
    db2 = np.where(db2_norm > k, (db2 / db2_norm) * k, db2)
    dw1 = np.where(dw1_norm > k, (dw1 / dw1_norm) * k, dw1)
    db1 = np.where(db1_norm > k, (db1 / db1_norm) * k, db1)
 
    print(f"New DW2: {np.max(dw2), np.min(dw2)}")
    return dw1, db1, dw2, db2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def grad_descent(x, y, w1, b1, w2, b2, epochs, alpha, k, file):
    one_hot_y = one_hot(y)
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)

        loss = cce(one_hot_y, a2)
        acc = accuracy(a2, y)

        dw1, db1, dw2, db2 = backward(x, one_hot_y, w2, a2, a1, z1, k)
        w1, b1, w2, b2 = update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
    
        print(f"Epoch: {epoch}")
        print(f"Loss: {loss}")
        print(f"Acc: {acc}\n")

    #save_model(file, w1, b1, w2, b2)
    print(f"Saved model!")

    return w1, b1, w2, b2

def model(x, y, epochs, alpha, k, file):

    try:
        w1, b1, w2, b2 = load_model(file)
        print(f"LOADED MODEL!")

    except FileNotFoundError:
        print(f"MODEL NOT FOUND. INIT NEW MODEL!")
        w1, b1, w2, b2 = init_params()
    
    w1, b1, w2, b2 = grad_descent(x, y, w1, b1, w2, b2, epochs, alpha, k, file)

    return w1, b1, w2, b2

if __name__ == "__main__":
    data = pd.read_csv('data/fashion-mnist_train.csv')
    data = np.array(data)

    X_train = data[:, 1:785].T / 255
    Y_train = data[:, 0].reshape(1, -1)

    alpha = .1
    epochs = 100
    k = .2
    file = 'models/nn.pkl'
    
    w1, b1, w2, b2 = model(X_train, Y_train, epochs, alpha, k, file)

    
    
    