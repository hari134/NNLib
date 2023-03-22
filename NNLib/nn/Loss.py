


import numpy as np
from collections import deque

class Loss():
    def __init__(self,parameters,_loss,_loss_logits_grad):
        self.parameters = parameters
        self._loss = _loss
        self._loss_logits_grad = _loss_logits_grad
        self.latest_grad = _loss_logits_grad

    def item(self):
        return self._loss
    
    def backward(self):
        for layer in reversed(self.parameters.layers):
            self.latest_grad = layer.backward(self.latest_grad)
    
        
class MSE:
    def __init__(self,parameters) -> None:
        self.parameters = parameters
        
    def __call__(self,input,target):
        _loss,_loss_logits_grad = _mse(input,target)
        return Loss(self.parameters,_loss,_loss_logits_grad)

def _mse(input,target):
    """input in nd numpy array where each row is one input sample

    Args:
        input (nd numpy array): the predicted values by the model
        target (nd numpy array): the actual values 

    Raises:
        ValueError: if the dimensions of input and target do not match
    """
        
    if(input.shape != target.shape):
        raise ValueError("Invalid dimensions of predicted value and target")
    
    _num_input = input.shape[0]
    _num_datapoints_per_input = input.shape[1]
    _loss = np.square(input-target).mean(axis=0)
    
    _loss_logits_grad = (input-target)/(_num_input)
    return _loss,_loss_logits_grad
    
    
    