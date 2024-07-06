from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

lr = LinearRegression()

data = pd.read_csv('multiple_linear_regression_dataset.csv')
data = np.array(data)

X_train = data[:, :2]
Y_train = data[:, 2].reshape(-1, 1)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
Y_train = ss.fit_transform(Y_train)

lr.fit(X_train, Y_train) 

Y_pred = lr.predict(X_train)

mse = mean_squared_error(Y_train, Y_pred)
r2 = r2_score(Y_train, Y_pred)

print(f'Muliple LR with sklearn LinearRegression')
print('weights', lr.coef_)
print('bias',  lr.intercept_)
print(f"MSE: {mse}")
print(f"R2: {r2}")