import numpy as np
import pickle
from preprocess import X_train, Y_train

def save_model(file, w1, b1, w2, b2):
    with open(file) as f:
        pickle.dump((w1, b1, w2, b2), f)
    return

def load_model(file):
    with open(file) as f:
        return pickle.load(f)

def init_params():
    w1 = np.random.randn(64, 3072) * np.sqrt(1/3072)
    b1 = np.zeros((64, 1))
    w2 = np.random.randn(10, 64) * np.sqrt(1/3072)
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return z > 0

def softmax(z):
    eps = 1e-10
    return np.exp(z + eps)/np.sum(np.exp(z + eps), axis = 0 , keepdims = True)

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1 # 64, 10000
    a1 = relu(z1) # 64, 10000
    z2 = np.dot(w2, a1) + b2 # 10, 10000
    a2 = softmax(z2) # 10, 10000
    return z1, a1, z2, a2

def prediction(a2):
    pred = np.argmax(a2, axis = 0, keepdims = True) #1, 10000
    return pred

def accuracy(a2, y):
    pred = prediction(a2) #1, 10000
    acc = np.sum(pred == y) / 10000 * 100 #averaging boolean output of prediction == labels, as a percentage
    return acc

def one_hot(y):
    one_hot_y = np.zeros((np.max(y) + 1, y.size))
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y

def cat_cross(one_hot_y, a):
    eps = 1e-10
    l = - np.sum(one_hot_y * np.log(a + eps)) / 10000
    return l

def backward(x, one_hot_y, a2, a1, w2):
    dz2 = a2 - one_hot_y # 10, 10000
    dw2 = np.dot(dz2, a1.T) / 10000
    db2 = np.sum(dz2, axis = 1, keepdims=True) / 10000
    dz1 = np.dot(w2.T, dz2) * relu_deriv(a1)
    dw1 = np.dot(dz1, x.T) / 10000
    db1 = np.sum(dz1, axis = 1, keepdims=True) / 10000 
    return dw2, db2, dw1, db1

def update(w2, b2, w1, b1, dw2, db2, dw1, db1, alpha):
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    return w1, b1, w2, b2


def gradient_descent(X_train, Y_train, w1, b1, w2, b2, epochs, alpha, file):
    one_hot_y = one_hot(Y_train)
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(X_train, w1, b1, w2, b2)
        
        l = cat_cross(one_hot_y, a2)
        acc = accuracy(a2, Y_train)

        dw2, db2, dw1, db1 = backward(X_train, one_hot_y, a2, a1, w2)
        w1, b1, w2, b2 = update(w2, b2, w1, b1, dw2, db2, dw1, db1, alpha)


        print(f"Epoch: {epoch}")
        print(f"Accuracy: {acc}%")
        print(f"Loss: {l}")

    save_model(file, w1, b1, w2, b2)
    return w1, b1, w2, b2

def model(X_train, Y_train, epochs, alpha, file):
    try:
        w1, b1, w2, b2 = load_model(file)
        print("model fuond, loading params!")
    except FileNotFoundError:
        print("model not found, initializing new params!")
        w1, b1, w2, b2 = init_params()
    
    w1, b1, w2, b2 = gradient_descent(X_train, Y_train, w1, b1, w2, b2, epochs, alpha, file)
    return w1, b1, w2, b2

if __name__ == "__main__":

    file = 'models/nncifar.pkl'
    params = model(X_train, Y_train, 1000, .01, file)



