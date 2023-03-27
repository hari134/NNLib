import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self,lr,epochs,optimizer="sgd"):
        self.params = {'w':0, 'b':0}
        self.lr = lr
        self.epochs = epochs
        if optimizer not in ["sgd","gd"]: raise("Invalid optimizer")
        self.optimizer = optimizer
    
    def fit(self,X,Y):
        self.input = {'X':X,'Y':Y}
        if(X.shape[0] != Y.shape[0]):
            raise("Dimensions of input and output must be same")
        match self.optimizer:
            case "sgd":
                self.__sgd(X.T,Y.T)
            case "gd":
                self.__gd(X.T,Y.T)
        
    def get_params(self):
        return self.params
    
    def set_params(self,w,b):
        self.params['w'] = w
        self.params['b'] = b
    
    def plot(self):
        x = np.linspace(0,16200,16200)
        y = self.params['w']*x + self.params['b']
        plt.plot(x,y)
        plt.scatter(self.input['X'],self.input['Y'])
        plt.show()
    
    def predict(self,x):
        return self.params['w']*x + self.params['b']

    def __sgd(self,x,y):
        for _ in range(self.epochs):
            index = np.random.randint(0,x.shape[0],1)
            x_s = np.take(x,index)
            y_s = np.take(y,index)
            y_pred = (self.params['w']*x_s) + self.params['b']
            self.params['w'] -= self.lr*2*np.sum(np.matmul((y_pred - y_s),x_s.T))
            self.params['b'] -= self.lr*2*np.sum(np.sum(y_pred-y_s))
    
    def __gd(self,x,y):
        size = x.shape[0]
        for _ in range(self.epochs):
            y_pred = (self.params['w']*x) + self.params['b']
            self.params['w'] -= self.lr*2*np.sum(np.matmul((y_pred - y),x.T))*(1/size)
            self.params['b'] -= self.lr*2*np.sum(np.sum(y_pred-y))*(1/size)     
    