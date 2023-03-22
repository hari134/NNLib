# Neural Networks and ML Algorithms

This library(NNLib) is a Python implementation of a simple neural network library, built using NumPy with a PyTorch-like syntax. The library currently supports the creation of linear layers and several activation functions commonly used in deep learning, such as ReLU, sigmoid, and softmax.

This library is intended to be used as a learning tool for those new to neural networks, rather than as a production-grade library. The code is implemented to run on the CPU only.

<strong>NOTE : </strong> <em>Please note that this library is a personal project that was built to test my understanding of neural network algorithms.It is not intended for use in production environments.</em>

## Installation

To use this library, you can clone this repository and install the required packages using pip:

```shell
git clone https://github.com/hari134/Neural_Networks_and_ML_Algorithms.git
cd Neural_Networks_and_ML_Algorithms
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
        
network = DenseNN(784, 128, 64, 10)                  #initializing the layer dimensions
criterion = nn.MSE(network.parameters())             #configuring MSE loss
optimizer = optim.SGD(network.parameters(), lr=0.01) #configuring SGD for optimization

#training
def train(epochs, x_train, y_train, optimizer):
    for epoch in range(epochs):
        running_loss = 0
        for i in range(60000):
            a0 = np.array([x_train[i]]).T
            y_true = np.array([y_train[i]]).T
            y_pred = network.forward(a0)
            loss = criterion(y_pred, y_true)         #computes the loss and returns a loss object
            loss.backward()                          #computes the gradients
            optimizer.step()                         #updates the parameters based on the computed gradients
            running_loss += loss.item()              #loss.item() returns scalar loss value
        
```


