import numpy as np
from math import sqrt
from abc import ABC, abstractmethod


class WeightInitializer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def initialize(self, *args):
        pass


class Xavier(WeightInitializer):
    """For softmax,sigmoid or no activation"""
    def __init__(self) -> None:
        pass
    
    def initialize(self, n_prev, n_curr):
        np.random.seed(0)
        lower, upper = -1/sqrt(n_prev), 1/sqrt(n_prev)
        weights = np.random.standard_normal(
            lower, upper, size=(n_curr, n_prev))
        return weights


class NormalizedXavier(WeightInitializer):
    """For softmax,sigmoid or no activation"""
    def __init__(self) -> None:
        pass
    
    def initialize(self):
        np.random.seed(0)
        lower ,upper = -6/sqrt(n_prev + n_curr) , 6/sqrt(n_prev + n_curr)
        weights = np.random.standard_normal(lower,upper,size = (n_curr,n_prev))
        return weights
     

class He(WeightInitializer):
    """For Relu and its variants"""
    def __init__(self) -> None:
        pass
        
    def initialize(self, n_prev, n_curr):
        weights = np.random.randn(n_curr, n_prev) * np.sqrt(2. / n_curr)
        return weights
    

