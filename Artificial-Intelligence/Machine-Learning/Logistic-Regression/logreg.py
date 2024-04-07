# 1 FEATURE, 1 TARGET (binary)

import pandas as pd
import numpy as np
import pickle

# SAVING MODEL
def save_model(w, b, filename):
    with open(filename, 'wb') as f:
        pickle.dump((w,b), f)

# LOADING MODEL
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# PARAMS
def params():
    w = np.random.rand(1)
    b = np.random.rand(1)
    return w,b


# FORWARD
def forward(x, w, b):
    z = np.dot(x, w) + b
    a = sigmoid(z)
    return a

# SIGMOID
def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a

# LOSS FUNC | CROSS ENTROPY LOSS
def log_loss(a, y):
    loss = -np.mean(y * np.log(a) + (1 - y) * np.log(1-a))
    return loss


# BACKPROP (COMPUTING GRADIENTS)
def back_prop(a, x, y):
    dw = np.mean((a - y) * x)
    db = np.mean((a - y))
    return dw, db

# UPDATE PARAMS
def update(dw, db, w, b, alpha):
    w = w - alpha * dw
    b = b - alpha * db
    return w, b


# GRADIENT DESCENT
def gradient_descent(x, y, epochs, alpha):

    model_filename = 'models/linreg.pkl'
    try:
        w, b = load_model(model_filename)
        print(f"Loaded model from {model_filename}")
    
    except FileNotFoundError:
        print("Model not found. Initializing new model!")
        w, b = params()

    for i in range(epochs):
        a = forward(x, w, b)
        
        loss = log_loss(a, y)

        print(f"Epoch: {i}")
        print(f"Loss: {loss}")

        dw, db = back_prop(a, x, y)
        w, b = update(dw, db, w, b, alpha)
    return w, b

if __name__ == "__main__":

    data = pd.read_csv("./Data/randomtrain.csv")
    data = np.array(data)

    x_train = data[:, 0].reshape(-1,1) # 200, 1
    y_train = data[:, 2].reshape(-1,1) # 200, 1


    epochs = 1000
    alpha = .0001

    w, b = gradient_descent(x_train, y_train, epochs, alpha)
    save_model(w, b, 'models/linreg.pkl')