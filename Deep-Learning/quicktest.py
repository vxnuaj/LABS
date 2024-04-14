import numpy as np

def init_params(dims):
  W = np.random.rand(1, dims)
  B = np.random.rand(1, 1)
  return W, B

def sigmoid(Z):
  A = 1 / (1 + np.exp(-Z))
  return A

def forward(X, W, B):
  Z = np.dot(W, X) + B
  A = sigmoid(Z)
  return A

def log_loss(Y, A):
    loss = np.mean( -Y * np.log(A) - (1 - Y) * np.log (1 - A))
    return loss

def back_prop(X, Y, W, B, A, alpha):
  dW = np.mean(X * (A - Y))
  dB = (A - Y)
  W = W - alpha * dW
  B = B - alpha * dB
  return W, B

def gradient_descent(X, Y, epochs, alpha, dims):
  W, B = init_params(dims)
  for epoch in range(epochs):
    A = forward(X, W, B)
    loss = log_loss(Y, A)
    W, B = back_prop(W, B, Y, X, A, alpha)

    print(f"Epoch: {epoch}")
    print(f"Loss: {loss}")

  return W, B
  

if __name__ == "__main__":
     
    X = np.array([[1, 2], [3, 4], [5, 6]])
    Y = np.array([[0],[1],[1]])
    gradient_descent(X, Y, 2000, .0001, 3)