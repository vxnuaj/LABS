import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ewa(data, beta):
    ewa_arr = np.zeros((data.shape[0], 2))
    V = 0
    for i in range(len(data)): # 0 -> len(data)
        V = (beta * V) + ((1 - beta) * data[i, 1])
        V = (V) / (1 - (beta ** (i + 1)))
        ewa_arr[i, 1] = V
        ewa_arr[i, 0] = i + 1
    return ewa_arr

if __name__ == "__main__":
    data = pd.read_csv('data/ewa.csv')
    data = np.array(data)


    ewa_arr = ewa(data, 0.2)

    print(ewa_arr)


    plt.plot(ewa_arr[:, 1], label = 'ewa')
    plt.plot(data[:, 1], label = 'data')
    plt.legend()
    plt.show()
