import numpy as np
import h5py as h5
import pickle

def load_model(filename):
    with open(filename, 'rb') as model:
        w, b = pickle.load(model)
        return w, b

def save_model(filename, w, b):
    with open(filename, 'wb') as model:
        pickle.dump((w, b), model)
    return


def load_data():
    f = h5.File('Artificial-Intelligence/Machine-Learning/AndrewNG-Notes/data/train_catvnoncat.h5')
    return f

def load_test_data():
    f = h5.File('Artificial-Intelligence/Machine-Learning/AndrewNG-Notes/data/test_catvnoncat.h5')
    return f

def init_params(features):
    w = np.random.rand(1, features)
    b = 0
    return w, b

def forward(w, b, x):
    z = np.dot(w, x) + b
    a = sigmoid(z)
    return a

def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a

def get_pred(a):
    if a > .5:
        pred = 1
    else:
        pred = 0
    return pred

def log_loss(a, y):
    eps = 1e-10
    loss = - np.mean(y * np.log(a + eps) + (1-y) * np.log(1-a + eps))
    return loss

def back_prop(x, y, w, b, a, alpha):
    dw = np.mean(a - y * x)
    db = np.mean(a - y)
    w = w - alpha * dw
    b = b - alpha * db
    return w, b

def gradient_descent(x, y, w, b, alpha, epochs):
    for epoch in range(epochs):
        a = forward(w, b, x)
        w, b = back_prop(x, y, w, b, a, alpha)
        loss = log_loss(a, y)

        print(f"Loss: {loss}")
        print(f"Epoch: {epoch}")
    return w, b

def model(x, y, features, alpha, epochs):

    try:
        w, b = load_model('Artificial-Intelligence/Machine-Learning/AndrewNG-Notes/models/lrc.pkl')
        print(f"Model found! Initializing!")
    except:
        print(f"Model not found! Initializing params!")
        w, b = init_params(features)

    w, b = gradient_descent(x, y, w, b, alpha, epochs)
    save_model('Artificial-Intelligence/Machine-Learning/AndrewNG-Notes/models/lrc.pkl', w, b)

    return

if __name__ == "__main__":
    f = load_data()

    X_train = np.array(f['train_set_x']) # DIMS: (209, 64, 64, 3) | (samples, width per sample, height per sample, per sample rgb matrices)
    Y_train = np.array(f['train_set_y']) # DIMS: (209,) | (Number of sample labels)

    X_train = X_train.reshape(209, -1) / 255 # DIMS: 209, 12288 | (samples, feature per sample)
    Y_train = Y_train.reshape(209, -1) # DIMS; 209, 1 | (total samples, label per sample)

    X_train = X_train.T
    Y_train = Y_train.T

    model(X_train, Y_train, 12288, .0002, 3000)
