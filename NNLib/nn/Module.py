from collections import deque

class LayerObj:
    def __init__(self,layers) -> None:
        self.layers = layers
    
    def forward(self,input):
        o = input
        for layer in self.layers:
            o = layer(o)
        return o


class Module:
    def __init__(self):
        self. __LAYERS = deque()

    def add_layer(self, layer):
        self.__LAYERS.append(layer)
        
    def parameters(self):
        return self.LayerObj
    
    def Sequential(self,layers):
        self.__LAYERS = deque()
        for layer in layers: 
            self.add_layer(layer)
        self.LayerObj = LayerObj(layers)
        return self.LayerObj
    

