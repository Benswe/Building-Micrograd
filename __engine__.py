import math 
import numpy as np
import matplotlib.pyplot as plt

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
            self.grad = 1 * out.grad # addition is easiest, just 1 * dl/dout
            other.grad = 1 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op="*")
        
        def _backward(): # update gradients for when backprop is called
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad

        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"{self.data}"
    
    def __radd__(self, other): # incase user wants to do dtype + val
        return self + other
    
    def __rmul__(self, other): # for reverse multiplication cases 
        return self * other
    

    def tanh(self):
        # sinh/cosh
        out = Value((np.sinh(self.data)/np.cosh(self.data)), _children=(self, ), _op="tanh")

        def _backward():
            # use the fact that the derivative of tanh(self) is (1-tanh**2)
            self.grad = 1-(out.data**2)
        out._backward = _backward
        return out 



 
val = Value(3)

v = val.tanh()
v._backward()
print(val.grad)