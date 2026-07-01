import math 
import numpy as np

"""
this is where we will define all of our operations that we can do 
For each operation we will also implement the backwards pass 
"""
class Value:
    def __init__(self, value, _children=(), _op=''):
        self.data = value
        self.grad = 0 # always start the gradient at 0
        self._backward = lambda: None # backward must be implemented for each operation
        self._prev = set(_children)
        self._op = _op


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op="+")


        def _backward(): 
            self.grad += 1 * out.grad # addition is easiest, just 1 * dl/dout
            other.grad += 1 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op="*")
        
        def _backward(): # update gradients for when backprop is called
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"{self.data}"
    
    def __radd__(self, other): # incase user wants to do dtype + val
        return self + other
    
    def __rmul__(self, other): # for reverse multiplication cases 
        return self * other
    
    def __pow__(self, x):
        assert isinstance(x, (int, float))
        out = Value(self.data ** x, (self, ), f"**{x}")
        def _backward():
            self.grad += x*(self.data ** (x-1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return(self * other**-1)


    def __neg__(self):
        return self * -1
    # e^self

    def __sub__(self, other):
        return self + (-other)
    
    
    def exp(self): # e^self
        out = Value(np.exp(self.data), _children=(self, ), _op = 'exp') 
        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    
    def log(self):
        out = Value(np.log(self.data), _children=(self, ), _op="log")

        def _backward():
            self.grad += (1/self.data) * out.grad
        
        out._backward = _backward
        return out



    def tanh(self):
        # sinh/cosh
        t = (np.exp(2 * self.data) - 1) / (np.exp(2 * self.data) + 1)
        out = Value(t, (self,), "tanh")
        def _backward():
            # use the fact that the derivative of tanh is (1-tanh**2)
            self.grad += (1-(t**2)) * out.grad
        out._backward = _backward
        return out 
    
    def backward(self): # use topo sort
        # we want to get all nodes and then call _backward() on them
        topo = []
        visited = set()
        def topo_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topo_sort(child) # recursive call on node
                topo.append(v)
        topo_sort(self)
        self.grad = 1 # gradient of self wrt to self is 1
        # we want to call _backward on the last layer first
        for node in reversed(topo):
            node._backward()



## lets build out an expression to test 
if __name__ == "__main__":
    # inputs
    x1 = Value(2.0)
    x2 = Value(1.0)
    a = x1+x2
    b = a.tanh()
    print(type(a.data))
    print(type(float(b.data))) # b.data should not be a int it should be a float
    b.backward()
    print(a.grad)
        
