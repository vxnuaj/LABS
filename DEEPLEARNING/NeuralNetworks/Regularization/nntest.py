import numpy as np
import pandas as pd
from nn import forward, accuracy, one_hot, load_model
from nn import cce

def test_nn(x, y, model):
    one_hot_y = one_hot(y)
    try:
        w1, b1, w2, b2 = load_model(model)
        print('LOADED MODEL!')
    except FileNotFoundError:
        print('NO MODEL FOUND!')

    _, _, _, a2 = forward(x, w1, b1, w2, b2)

    loss = cce(one_hot_y, a2)
    acc = accuracy(a2, y)

    print(f"Model: {model}")
    print(f"Loss: {loss}")
    print(f"Acc: {acc}")


if __name__ == "__main__":
    data = pd.read_csv('data/fashion-mnist_test.csv')
    data = np.array(data)

    X_test = data[:, 1:785].T / 255
    Y_test = data[:, 0].reshape(1, -1)

    model = 'models/nn.pkl'

    test_nn(X_test, Y_test, model)


'''

def cce(one_hot_y, a):
    eps = 1e-10
    l1_norm = (lambd * (np.sum(np.abs(w2)) + np.sum(np.abs(w1))) / y.size
    l = - np.sum(one_hot_y * np.log(a + eps)) / one_hot_y.shape[1]
    reg_l = l + l1_norm
    return l    

def backward(x, one_hot_y, w2, a2, a1, z1):
    dz2 = a2 - one_hot_y
    dw2 = (np.dot(dz2, a1.T) + (lambd * np.sign(w2)) / one_hot_y.shape[1]
    db2 = np.sum(dz2, axis = 1, keepdims=True) / one_hot_y.shape[1]
    dz1 = np.dot(w2.T, dz2) * leaky_relu_deriv(z1)
    dw1 = (np.dot(dz1, x.T) + (lambd * np.sign(w)) / one_hot_y.shape[1]
    db1 = np.sum(dz1, axis = 1, keepdims=True) / one_hot_y.shape[1]
    return dw1, db1, dw2, db2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

'''