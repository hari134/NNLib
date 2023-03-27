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
        self.weights = self.__WEIGHT_INITIALIZERS["relu"].initialize(self.n_prev, self.n_curr)

    @abstractmethod
    def __init__(self, n_prev, n_curr, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, input):
        pass

    @abstractmethod
    def backward(self,*args):
        pass

    @abstractmethod
    def update(self,*args):
        pass


class Linear(Layer):
    def __init__(self, n_prev, n_curr):
        self.type = "linear"
        self.n_prev = n_prev
        self.n_curr = n_curr
        self.grad = None
        self.bias = np.zeros((n_curr, 1))
        self.set_weights()

    def __call__(self, input):
        self.input = input
        self.output = np.dot(self.weights, input) + self.bias
        return self.output

    def _compute_grad_w(self, latest_grad):
        self.grad_w = np.dot(latest_grad, (self.input).T)

    def _compute_grad_b(self, latest_grad):
        self.grad_b = latest_grad.mean(axis=1).T.reshape(latest_grad.shape[0],1)

    def backward(self, latest_grad):
        self.input = self.input.mean(axis=1).reshape(self.input.shape[0],1)
        self._compute_grad_w(latest_grad)
        self._compute_grad_b(latest_grad)
        self.grad = np.dot(self.weights.T, latest_grad)
        return self.grad

    def update(self, lr):
        # print(self.grad_w.shape)
        # print(self.grad_b.shape)
        # print(self.bias.shape)
        # print(self.weights.shape)

        self.weights -= lr * self.grad_w
        self.bias -= lr * self.grad_b


class Conv2d(Layer):
    def __init__(self, in_channel, out_channel, kernel_size):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
    

class MaxPooling2d(Layer):
    def __init__(self):
        pass


class Dropout(Layer):
    def __init__(self, *args):
        raise NotImplementedError
    

class Flatten(Layer):
    def __init__(self, *args):
        raise NotImplementedError