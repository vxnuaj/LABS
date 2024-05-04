import numpy as np
import pandas as pd
import sys


from nue.models import nn


if __name__ == "__main__":

    data = pd.read_csv('data/mnist_train.csv')
    data = np.array(data) # 60000, 785

    Y_train = data[:, 0].T.reshape(1, -1)# 1, 60000
    X_train = data[:, 1:786].T / 255 # 784, 60000

    model = nn.NN(X_train, Y_train, 784, 10, 32, .1, 1000)

    model.model()

    '''params = model.init_params()

    #print(w1.shape) # 32, 784
    #print(b1.shape) # 32, 1
    #print(w2.shape) # 10, 32
    #print(b2.shape) # 10, 1

    pred = model.forward()

    #print(pred.shape) # 10, 60000

    z1, a1, z2 =  model.outputs
    
    #print(z1.shape) # 32, 60000
    #print(a1.shape) # 32, 60000
    #print(z2.shape) # 10, 60000

    one_hot_y = model.one_hot()
    
    #print(one_hot_y.shape) # 10, 60000

    
    l = model.cat_cross_entropy()

    # print(l)


    dw2, db2, dw1, db1 = model.backward()

    #print(f"dw2 {dw2.shape}") # 10, 32
    #print(f"db2 {db2.shape}") # 10, 1
    #print(f"dw1 {dw1.shape}") # 32, 784
    #print(f"db1 {db1.shape}") # 32, 1

    # 10, 60000 @ 60000, 32 -> 10, 32 | when calculaing dw2
    # 32, 10 @ 10, 60000 -> 32, 60000 | when calculating dz1
    # 32, 60000 @ 60000, 784 -> 32, 784 | when calculating dw1

    #print(model.gradients[0].shape) # 32, 784

    w1, b1, w2, b2 = model.update()

    #print(w1.shape) # 32, 784
    #print(b1.shape) # 32, 1
    #print(w2.shape) # 10, 32
    #print(b2.shape) # 10 ,1

    w1, b1, w2, b2 = model.gradient_descent()

    print(w1.shape) # 32, 784
    print(b1.shape) # 32, 1
    print(w2.shape) # 10, 32
    print(b2.shape) # 10 ,1'''