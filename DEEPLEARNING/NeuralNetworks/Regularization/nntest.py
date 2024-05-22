import numpy as np
import pandas as pd
from l2nn import forward, accuracy, one_hot, load_model
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
    print(f"W Max: {np.max(w2)}")
    print(f"W Min: {np.min(w2)}\n")


if __name__ == "__main__":
    data = pd.read_csv('data/fashion-mnist_test.csv')
    data = np.array(data)

    X_test = data[:, 1:785].T / 255
    Y_test = data[:, 0].reshape(1, -1)

    model1 = 'models/orignn.pkl'
    model2 = 'models/l2nn.pkl'

    test_nn(X_test, Y_test, model1)
    test_nn(X_test, Y_test, model2)