# Neural Networks and ML Algorithms

This library(NNLib) is a Python implementation of a simple neural network, built using NumPy. The library currently supports the creation of linear layers and several activation functions commonly used in deep learning, such as ReLU, sigmoid, and softmax.

This library is intended to be used as a learning tool for those new to neural networks, rather than as a production-grade library. The code is implemented to run on the CPU only.

<strong>NOTE : </strong> <em>Please note that this library is a personal project that was built to test my understanding of neural network algorithms.It is not intended for use in production environments.</em>

## Installation

To use this library, you can clone this repository and install the required packages using pip:

```shell
git clone https://github.com/hari134/Neural_Networks_and_ML_Algorithms.git
cd your_repository
pip install -r requirements.txt
```
Usage examples can be found in the Examples directory.

<em>Code to define a 3 layer dense neural network, loss and optimization functions</em>

```python

import NNLib.nn as nn
import NNLib.optim as optim

# defining the neural network

class DenseNN(nn.Module):
    def __init__(self, d0, d1, d2, d3):
        self.Layers = self.Sequential(
            [
                nn.Linear(d0, d1),
                nn.Sigmoid(),
                nn.Linear(d1, d2),
                nn.Sigmoid(),
                nn.Linear(d2, d3),
                nn.Softmax(),
            ]
        )

    def forward(self, z0):
        o = self.Layers.forward(z0)
        return o
        
network = DenseNN(784, 128, 64, 10)
criterion = nn.MSE(network.parameters())
optimizer = optim.SGD(network.parameters(), lr=0.01)
        
```


