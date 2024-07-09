import pandas as pd
from nue.models import LinearRegression 
from nue.preprocessing import io, data_split as ds
from nue.metrics import mse, r2_score


# our data is of ( samples, features ). we have 303 samples, 14 features

data = io.csv_to_numpy('data/linear_regression_dataset.csv')
train, test = ds.train_test_split(data, .8)
X_train, Y_train = ds.x_y_split(train, y_col = 'last')
X_test, Y_test = ds.x_y_split(test, y_col = 'last')

X_train = X_train

alpha = .001
epochs = 20000

model = LinearRegression()

model.train(X_train, Y_train, alpha, epochs, verbose = False, metric_freq=500)

model.test(X_test, Y_test)


model.metrics(mode = 'both')