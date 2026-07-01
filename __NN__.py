import numpy as np
import math
import random
from __engine__ import Value





class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b= Value(random.uniform(-1, 1))

    def __call__(self, x):
        # dot product between w and x 
        wxb = np.dot(x, self.w)
        wxb += self.b # add the bias 
        out = wxb.tanh() # restrict between -1 and 1 
        return out
    def parameters(self):
        return self.w + [self.b] # parameters


class Layer:
    def __init__(self, nin, nout):
        self.layer = [Neuron(nin) for _ in range(nout)]


    def __call__(self, x):
        out = [n(x) for n in self.layer]
        return out

    def parameters(self):
        params = [] 
        for neuron in self.layer:
            # add each vector of paramters to the params vector 
            params.extend(neuron.parameters()) 
        return params


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        # input trails output so use zip
        self.MLP = [Layer(input, out) for input, out in zip(sizes, sizes[1:])]

    def __call__(self, x):
        prev_x = x
        for layer in self.MLP:
            x = layer(prev_x)
            prev_x = x
        return prev_x[0]

    
    def parameters(self):
        params = []
        for layer in self.MLP:
            params.extend(layer.parameters())
        return params



# classification problem
if __name__ == "__main__":
    
    xs = [
        [0, 5, 3],
        [8, 2, 5],
        [2, 1, 0]
    ]


    model = MLP(3, [10, 10, 1]) # input layer of 5, then two intermediate layers of 4 
    ys = [0.92, -0.1, 0.58391]
    for _ in range(100):
        # forward pass
        preds = [model(x) for x in xs]
        loss = sum([((pred - y)**2) for y,pred in zip(ys, preds)])/len(preds)
        print(loss)
        #backward pass
        loss.backward()

        # Gradient descent 
        for p in model.parameters():
            p.data += -0.1 * p.grad
            p.grad = 0
    print(preds)