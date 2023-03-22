#TODO add provision to specify number of inputs in each step

class SGD:
    __LAYER_TYPES = ["linear"]
    def __init__(self,parameters,lr,*args,**kwargs) -> None:
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for layer in self.parameters.layers:
            if layer.type in self.__LAYER_TYPES:
                layer.update(self.lr)

                
