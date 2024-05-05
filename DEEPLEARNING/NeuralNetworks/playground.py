import numpy as np

x = np.array(([2, 3, 1])) # (1, 3) (1 sample, 3 features)

def init_params():
    w1 = np.random.randn(5, 3) * np.sqrt(1/3)
    b1 = np.random.randn(5, 1)
    w2 = np.random.randn(5, 5) * np.sqrt(1/3)
    b2 = np.random.randn(5, 1)
    w3 = np.random.randn(3, 5) * np.sqrt(1/3)
    b3 = np.random.randn(3, 1)
    w4 = np.random.randn(1, 3) * np.sqrt(1/3)
    b4 = np.random.randn(1, 1)
    return w1, b1, w2, b2, w3, b3, w4, b4

def sigmoid(z):
    eps = 1e-10
    a = 1/(1+np.exp(-z + eps))
    return a

def forward(x, w1, b1, w2, b2, w3, b3, w4, b4):
    z1 = np.dot(w1, x) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = sigmoid(z3)
    z4 = np.dot(w4, a3) + b4
    a4 = sigmoid(z4)
    return a1, a2, a3, a4

if __name__ == "__main__":
    w1, b1, w2, b2, w3, b3, w4, b4 = init_params()
    a1, a2, a3, a4 = forward(x,w1, b1, w2, b2, w3, b3, w4, b4)

    print(a4)