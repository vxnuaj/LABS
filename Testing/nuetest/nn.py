import numpy as np
import pandas as pd
from nue.models import NN 

data = pd.read_csv("data/fashion-mnist_train.csv")
data = np.array(data)


X_train = data[:, 1:786].T / 255
Y_train = data [:, 0].T.reshape(-1, 60000)

print(Y_train.shape)

#model = NN(X_train, Y_train, 784, 10, 30, .1, 500)

#model.model()
