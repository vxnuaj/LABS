import numpy as np
import pandas as pd

data = pd.read_csv('data/linear_regression_dataset2.csv')
data = np.array(data)

X_train = data[:, 0:4].T # (3, 500) 
Y_train = data[:, 4].T.reshape(-1, 2000)

X_train_scaled = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) - np.min(X_train, axis=0))

print(X_train.shape)

from nue import linreg

lr = linreg.LinearRegression(X_train, Y_train, 4, .001, 10000)

w, b = lr.model()