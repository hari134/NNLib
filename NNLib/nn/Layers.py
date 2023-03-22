import Weights
from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    __WEIGHT_INITIALIZERS = {
        "relu": Weights.He(),
        "sigmoid": Weights.Xavier(),
        "softmax": Weights.He(),
    }

    def set_weights(self):
        self.weights = self.__WEIGHT_INITIALIZERS["relu"].initialize(self.n_prev,self.n_curr)
                

    @abstractmethod
    def __init__(self, n_prev, n_curr, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, input):
        pass


    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def update(self):
        pass


class Linear(Layer):
    def __init__(self, n_prev, n_curr):
        self.type = "linear"
        self.n_prev = n_prev
        self.n_curr = n_curr
        self.grad = None
        self.bias = np.zeros((n_curr,1))
        self.set_weights()
        

    def __call__(self, input):
        self.input = input
        self.output = np.dot(self.weights, input) + self.bias
        return self.output

    def _compute_grad_w(self,latest_grad):
        self.grad_w = np.dot(latest_grad,(self.input).T)

    def _compute_grad_b(self,latest_grad):
        self.grad_b = latest_grad

    def backward(self,latest_grad):
        self._compute_grad_w(latest_grad)
        self._compute_grad_b(latest_grad)
        self.grad = np.dot((self.weights).T,latest_grad)
        return self.grad

    def update(self,lr):
        self.weights -= lr * self.grad_w
        self.bias -= lr * self.grad_b

        
class Conv2d(Layer):
    def __init__(self, *args):
        raise NotImplementedError
