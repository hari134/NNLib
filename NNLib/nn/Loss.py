import numpy as np


class Loss:
    def __init__(self, parameters, _loss, _loss_grad):
        self.parameters = parameters
        self._loss = _loss
        self._loss_grad = _loss_grad
        self.latest_grad = _loss_grad

    def item(self):
        return self._loss

    def backward(self):
        for layer in reversed(self.parameters.layers):
            self.latest_grad = layer.backward(self.latest_grad)


class MSE:
    def __init__(self, parameters) -> None:
        self.parameters = parameters

    def __call__(self, input, target):
        _loss, _loss_grad = _mse(input, target)
        return Loss(self.parameters, _loss, _loss_grad)


def _mse(predicted, target):
    """input in nd numpy array where each row is one input sample

    Args:
        predicted (nd numpy array): the predicted values by the model
        target (nd numpy array): the actual values 

    Raises:
        ValueError: if the dimensions of input and target do not match
    """

    if predicted.shape != target.shape:
        raise ValueError("Invalid dimensions of predicted value and target")

    _num_input = predicted.shape[0]
    _num_datapoints_per_input = predicted.shape[1]
    _loss = np.square(predicted - target).mean()
    
    _loss_grad = ((predicted - target) / _num_input).mean(axis=1).reshape(_num_input,1)
    return _loss, _loss_grad
