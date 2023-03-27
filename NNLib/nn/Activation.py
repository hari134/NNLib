import numpy as np
from abc import ABC, abstractmethod



class BaseActivation(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, input):
        pass

    @abstractmethod
    def backward(self):
        pass


class Sigmoid(BaseActivation):
    def __init__(self):
        self.type = "sigmoid"

    def __call__(self, input):
        self.activated = 1 / (1 + np.exp(-input))
        return self.activated

    def backward(self, latest_grad):
        self.activated = self.activated.mean(axis=1).reshape(self.activated.shape[0],1)
        self.grad = latest_grad * (self.activated * (1 - self.activated))
        return self.grad


class Relu(BaseActivation):
    def __init__(self):
        self.type = "relu"

    def __call__(self, input):
        self.output = np.maximum(0, input)
        return self.output

    def backward(self, latest_grad):
        self.output = self.output.mean(axis=1).reshape(self.output.shape[0],1)
        self.grad = latest_grad * (1 * (self.output > 0))
        return self.grad


class Softmax(BaseActivation):
    def __init__(self):
        self.type = "softmax"

    def __call__(self, input):
        self.input = input
        y = np.exp(input - input.max())
        self.output = y / np.sum(y, axis=0)
        return self.output

    def backward(self, latest_grad):
        self.input = self.input.mean(axis=1).reshape(self.input.shape[0],1)
        x = self.input
        exps = np.exp(x - x.max())
        self.grad = latest_grad * (exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0)))
        return self.grad
