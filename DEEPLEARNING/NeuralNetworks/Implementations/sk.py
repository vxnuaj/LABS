from sklearn.linear_model import LinearRegression
import sklearn.metrics as skm
import pandas as pd
import numpy as np

data = pd.read_csv("./Data/toy.csv")
data = np.array(data)

x = data[:, :1]
y = data[:, 1]

x_reshaped = x.reshape(-1, 1)
y_reshaped = y.reshape(-1, 1)

reg = LinearRegression().fit(x_reshaped, y_reshaped)
y_pred = reg.predict(x_reshaped)

loss = skm.mean_absolute_error(y_reshaped, y_pred)

print("Scikit-learn slope:", reg.coef_)
print("Scikit-learn intercept:", reg.intercept_[0])
print("Mean square Error:", loss)
