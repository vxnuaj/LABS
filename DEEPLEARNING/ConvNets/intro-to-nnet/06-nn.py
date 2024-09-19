'''
Neural Network Class.

Assumes multi-class classification and hidden layer activation to be leaky relu

Input dims: (n_samples, n_features).T
Output dims: (n_samples, n_preds)
'''

import jax as jx
import jax.numpy as jnp
from jax import nn as jnn

from nue.preprocessing import x_y_split, csv_to_numpy, train_test_split, one_hot

import random
import time as t
from functools import partial
from termcolor import colored

class NN:
   
    eps = 1e-10 
    
    def __init__(self, train_verbose = False, test_verbose = False, seed = None):
        self.train_verbose = train_verbose
        self.test_verbose = test_verbose
        self.__key = None
        self.seed = seed
    
    def train(self, X_train, Y_train, nn_capacity:dict, alpha = .01, epochs = 250, Y_onehot = None, return_params = True, save_params = False): # save params to save parameters to directory, as pkl file.
        self.X_train = X_train
        self.Y_train = Y_train
        self.nn_capacity = nn_capacity
        self.alpha = alpha
        self.epochs = epochs
        self.Y_onehot = Y_onehot
        self.return_params = return_params

        self._init_internals() 
        self._init_params()
        self._gradient_descent() 
   
        if self.return_params:
            return self.weights, self.bias 
     
    def _init_params(self):
        
        if self.train_verbose:
            print(colored('Initializing Parameters', 'green', attrs = ['bold']))
        for layer, neurons in list(self.nn_capacity.items())[1:]:
            self.weights[layer - 1] = jx.random.normal(self.__key, shape = (neurons, list(self.nn_capacity.values())[layer - 1])) * jnp.sqrt(2 / list(self.nn_capacity.values())[layer - 1]) # Xavier Init.
            self.bias[layer - 1] = jnp.zeros(shape = (neurons, 1))
     
    def _gradient_descent(self):
        
        print(colored(f'Beginning Training, for {self.epochs} epochs, in 3 seconds.\n', 'green', attrs = ['bold']))

        for i in range(3):
            print(colored(i, 'green', attrs=['bold']))
            t.sleep(1)
       
        for epoch in range(self.epochs): 
            self._forward()
            if self.train_verbose:
                print(f"EPOCH -- {epoch + 1} | ACCURACY -- {self._accuracy()}% | CROSS ENTROPY LOSS -- {self._cross_entropy()}") 
            self._backward() 
            self._update_params()
     
    def _forward(self): 
        
        for layer in self.__layers_idxs:
            if layer == 1: 
                self.preactivations[layer - 1] = jnp.dot(self.weights[layer - 1], self.X_train) + self.bias[layer - 1]
            else:
                self.preactivations[layer - 1] = jnp.dot(self.weights[layer - 1], self.activations[layer - 2]) + self.bias[layer - 1]
                
            if layer != len(self.__layers_idxs):
                self.activations[layer - 1] = self._Leaky_ReLU(self.preactivations[layer - 1])
            else:
                self.activations[layer - 1] = self._softmax(self.preactivations[layer - 1]) 

    def _backward(self):
        for layer in reversed(self.__layers_idxs):
            if layer == self.__layers_idxs[-1]:
                self.grad_preacts[layer - 1] = self.activations[layer - 1] - self.Y_onehot
                self.grad_weights[layer - 1] = jnp.dot(self.grad_preacts[layer - 1], self.activations[layer - 2].T) * ( 1 / self.Y_train.size )
                self.grad_bias[layer - 1] = jnp.sum(self.grad_preacts[layer - 1], axis = 1, keepdims = True) * ( 1 / self.Y_train.size )
            else:
                self.grad_preacts[layer - 1] = jnp.dot(self.weights[layer].T, self.grad_preacts[layer]) * self._grad_Leaky_ReLU(self.preactivations[layer - 1])
                self.grad_weights[layer - 1] = jnp.dot(self.grad_preacts[layer - 1], self.X_train.T)  * ( 1 / self.Y_train.size )
                self.grad_bias[layer - 1] = jnp.sum(self.grad_preacts[layer - 1], axis = 1, keepdims = True) * ( 1 / self.Y_train.size )

    def _update_params(self):
        for layer in reversed(self.__layers_idxs):
            self.weights[layer - 1] -= self.alpha * self.grad_weights[layer - 1]
            self.bias[layer - 1] -= self.alpha * self.grad_bias[layer - 1]
           
    def _Leaky_ReLU(self, z):
        return jnp.maximum(.01 * z, z)  
   
    def _grad_Leaky_ReLU(self, z):
        return jnp.where(z > 0, 1, .01 )
    
    def _softmax(self, z):
       return jnp.exp(z + self.eps) / jnp.sum(jnp.exp(z + self.eps), axis = 0, keepdims = True) 
   
    def _cross_entropy(self):
        return - jnp.sum(self.Y_onehot * jnp.log(self.activations[-1] + self.eps)) * ( 1 / self.Y_train.size)

    def _accuracy(self):
        return jnp.mean(self.Y_train == jnp.argmax(self.activations[-1], axis = 0, keepdims = True)) * 100
    
    def _init_internals(self):
        
        self.__layers_idxs = list(self.nn_capacity.keys())[1:]
        
        self.preactivations = [0 * l for l in self.__layers_idxs] # list of jax.ndarray's
        self.activations = [0 * l for l in self.__layers_idxs] # list of jax.ndarray's
        self.weights = [0 * l for l in self.__layers_idxs] # list of jax.ndarray's
        self.bias = [0 * l for l in self.__layers_idxs] # list of jax.ndarray's 
        self.grad_preacts = [0 * l for l in self.__layers_idxs]
        self.grad_weights = [0 * l for l in self.__layers_idxs] # list of jax.ndarray's
        self.grad_bias = [0 * l for l in self.__layers_idxs] # list of jax.ndarray's

    def _debug(self):
        print()
        print(f"Output grad_preacts: {self.grad_preacts[1].shape} | Output preacts: {self.preactivations[1].shape}")
        print(f"Output grad_weights: {self.grad_weights[1].shape} | output weights: {self.weights[1].shape}")
        print(f"Output grad_bias: {self.bias[1].shape} | output bias: {self.bias[1].shape}")
        print()
        print(f"Hidden grad_preacts: {self.grad_preacts[0].shape} | Hidden preacts: {self.preactivations[0].shape}")
        print(f"Hidden grad_weights: {self.grad_weights[0].shape} | Hidden weights: {self.weights[0].shape}")
        print(f"Hidden grad_bias: {self.bias[0].shape} | Hidden bias: {self.bias[0].shape}")

    @property
    def Y_onehot(self):
        return self._Y_onehot

    @Y_onehot.setter
    def Y_onehot(self, Y_onehot):
        if Y_onehot is None:
            print(colored("No One Hot Encoding found. One Hot Encoding labels...", "green", attrs = ['bold']))
            Y_onehot = jnn.one_hot(self.Y_train, num_classes = jnp.max(self.Y_train) + 1, axis = 0)
            if len(Y_onehot.shape) == 3:
                Y_onehot = jnp.squeeze(Y_onehot, axis = 2)
        self._Y_onehot = Y_onehot

    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, seed):
        if seed is None:
            self._seed = random.randint(0, 1000)
        else:
            self._seed = seed
        self.__key = jx.random.key(self._seed)

if __name__ == "__main__":

    train_data = csv_to_numpy('data/fashion-mnist_train.csv')
    X_train, Y_train = x_y_split(train_data, y_col = 'first') 
    X_train = X_train.T / 255

    epochs = 100
    alpha = .01
    nn_capacity = {
       0:784, 
       1: 64,
       2: 10 
    }

    nnet = NN(train_verbose = True, seed = 1)
    nnet.train(X_train, Y_train, epochs = epochs, alpha = alpha, nn_capacity=nn_capacity)

'''
ON COMPILATION

- Lower level languages, like C++ or Rust, rely on static compilation to optimize code. This assumes static structures and sizes of variables.
- Higher level languages, like Python, rely on dynamic compilation, where code is compiled (optimized) assuming that variables defined in the program can change.
- Static compilation is more optimized, dynamic compilations sacrifice optimization for ease of use.

ON JAX -- JIT COMPILING

- Jax is designed with a purely functional programming paradigm -- no inplace ops
- Static shapes are known at compile time, they remain constant. Dynamic shapes are not known at compile time. They are variables that are prone to changing
- JIT compiling transforms functions into machine code using XLA, to optimize computations
    - XLA (acceleraged linalg), is the compiler backend for Jax, which is optimized for static compilation.
    - Static compilation, needs to know shapes beforehand, DYNAMIC SHAPES cannot be fully optimized.
- Just use static_argnums to predefine static shapes that might be considered dynamic by jax, so jax can treat them as static, lol

    # in the end, got : ValueError: Non-hashable static arguments are not supported. An error occurred while trying to hash an object of type <class 'jaxlib.xla_extension.ArrayImpl'>, 10. The error was: TypeError: unhashable type: 'jaxlib.xla_extension.ArrayImpl'
    # won't bother building this out, instead will use jax.nn.one_hot | WILL BE VALUABLE TO LEARN AFTER TORCH THOUGH.
    @staticmethod
    @partial(jit, static_argnums = (1,)) # treats the second arg, `n_classes` as a static value rather than a dynamic value, so jax can propertly jit compile.
    def _one_hot(Y_train, n_classes):
        n_samples = Y_train.size
        Y_onehot = lax.full(shape = (n_classes, n_samples), fill_value = 0, dtype = jnp.float32)
        Y_onehot = Y_onehot.at[Y_train, jnp.arange(n_samples)].set(1) # needs to be jit compiled, in order to run in place.
        return Y_onehot
'''