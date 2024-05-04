import numpy as np
import pandas as pd
from nue import nn

data = pd.read_csv("data/fashion-mnist_train.csv")
data = np.array(data)

print(data.shape)

X_train = data[:, 1:786].T / 255
Y_train = data [:, 0].T.reshape(-1, 60000)

model = nn.NN(X_train, Y_train, 784, 10, 30, .01, 500)

model.model()
