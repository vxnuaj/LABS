import numpy as np
import pandas as pd
import pickle

def unpickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f, encoding = "bytes")
    

data = unpickle('data/data_batch_1') #dict_keys([b'batch_label', b'labels', b'data', b'filenames'])

X_train = data[b'data'].T / 255 # 3072, 10000 (features, samples)

Y_train = data[b'labels'] # 1, 10000 (labels per sample, samples)
Y_train = np.array(Y_train).reshape(-1, 10000)