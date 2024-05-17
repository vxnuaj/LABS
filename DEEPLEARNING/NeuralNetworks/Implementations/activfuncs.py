import numpy as np

def sigmoid(z):
    eps = 1e-10
    return 1 / 1 + np.exp(-z + eps)

def sigmoid_deriv(z):
    eps = 1e-10
    return sigmoid(z) * (1 - sigmoid(z + eps))

def leaky_relu(z):
    return np.where(z > 0, z, .01 * z)

def leaky_relu_deriv(z):
    return np.where(z > 0, 1, .01)

def softmax(z):
    eps = 1e-10
    a = np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims = True)
    return a

def selu(z):
    eps = 1e-10
    return np.where(z > 0, 1.0505 * z, 1.0505 * (1.6732 * (np.exp(z + eps) - 1)))
    
def selu_deriv(z):
    eps = 1e-10
    return np.where(z > 0, 1.0505, 1.0505 * (1.6732 * np.exp(z + eps)))

def swish(z, beta = .01):
    eps = 1e-10
    return z * (1/(1+np.exp(beta * -z + eps)))

def swish_deriv(z, beta = 1):
    return swish(z, beta) + sigmoid(z) * (1 - swish(z, beta))

def elu(z, alpha = .01):
    eps = 1e-10
    return np.where(z > 0, z, alpha * (np.exp(z + eps) - 1))

def elu_deriv(z, alpha = .01):
    eps = 1e-10
    return np.where(z > 0, 1, alpha * np.exp(z + eps))

