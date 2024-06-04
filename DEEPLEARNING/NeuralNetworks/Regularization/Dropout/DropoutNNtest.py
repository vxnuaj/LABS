import numpy as np
import pandas as pd
from DropoutNN import predict, accuracy, one_hot, load_model, cce

def test_nn(x, y, model):
    one_hot_y = one_hot(y)
    try:
        w1, b1, w2, b2 = load_model(model)
        print('LOADED MODEL!')
    except FileNotFoundError:
        print('NO MODEL FOUND!')

    _, _, _, a2 = predict(x, w1, b1, w2, b2)

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

    model = 'models/dropoutNN.pkl'

    test_nn(X_test, Y_test, model)