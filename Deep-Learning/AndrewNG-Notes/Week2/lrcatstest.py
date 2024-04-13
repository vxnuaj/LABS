import numpy as np
from logregcats import load_model, load_test_data, forward, log_loss, get_pred


def test(x, y, w, b):
    correct = 0
    for i in range(len(y)):
        a = forward(w, b, x[:, i])
        loss = log_loss(a, y[i])

        pred = get_pred(a)

        print(f"Sample: {i}")
        print(f"Loss: {loss}")
        print(f"Prediction: {pred}")
        print(f"Real Val: {y[i]}")

        correct += (pred == y[i])
    
    accuracy = correct / len(y) * 100
    print(f"Accuracy: {accuracy}%")

if __name__ == "__main__":
    f = load_test_data()
    X_test = np.array(f['test_set_x']) # samples, height, width, rgb channels
    Y_test = np.array(f['test_set_y']) # sample labels,

    X_test = X_test.reshape(50, -1) / 255 # samples, features
    Y_test = Y_test.reshape(50, -1)

    X_test = X_test.T # features, samples

    w, b = load_model('Artificial-Intelligence/Machine-Learning/AndrewNG-Notes/models/lrc.pkl')

    test(X_test, Y_test, w, b)

