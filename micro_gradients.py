"""
A simple implementation of a micro-gradient system for automatic differentiation.
This code defines a Value class that represents a scalar value and its gradient.
The Value class supports basic arithmetic operations (addition, multiplication, power) and activation functions (ReLU, tanh, exp).
The backward method computes the gradients for all values in the computational graph using reverse-mode automatic differentiation.

Learning resources:
- "Micrograd: A tiny autograd engine in Python" by @Andrej_Karpathy: https://github.com/karpathy/micrograd
"""

import numpy as np 


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data # the scalar value
        self._prev = set(_children) # the set of parent nodes in the computational graph
        self._op = _op # the operation that produced this value (for visualization/debugging)
        self._backward = lambda: None # the function to compute gradients for this node
        self.label = label # optional label for visualization/debugging
        self.grad = 0.0 # the gradient of this value with respect to some loss (initialized to 0)
    
    # addition 
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+') 
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    # multiplication
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*') 
        def _backward(): 
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    # power
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supports int/float powers"
        out = Value(self.data ** other, (self,), f'**{other}') 
        def _backward(): 
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    # ReLU activation
    def relu(self):
        out = Value(self.data if self.data > 0 else 0, (self,), 'ReLU')
        def _backward(): 
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    # tanh activation
    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
    
    # exponential function
    def exp(self):
        e = np.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out
    
    # backward pass to compute gradients
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
    
    def __repr__(self):  
        return f"Value(data={self.data}, grad={self.grad})"

# example 
if __name__ == "__main__":
    a = Value(2.0, label='a')
    b = Value(3.0, label='b')
    c = a * b # 6
    d = a + b # 5
    e = c * d  # 30
    e.backward()
    
    # a: 21.0, b: 16.0, c: 5.0, d: 6.0, e: 1.0
    print(f"a: {a.grad}, b: {b.grad}, c: {c.grad}, d: {d.grad}, e: {e.grad}")
    
    # trace 
    # a = 2.0, b = 3.0
    # c = a * b = 6.0
    # d = a + b = 5.0
    # e = c * d = 30.0
    # e backward -> de/dc = d = 5.0, de/dd = c = 6.0
    # c backward -> dc/da = b = 3.0, dc/db = a = 2.0 
    # d backward -> dd/da = 1.0, dd/db = 1.0
    # a.grad = de/dc * dc/da + de/dd * dd/da = 5.0 * 3.0 + 6.0 * 1.0 = 21.0
    # b.grad = de/dc * dc/db + de/dd * dd/db = 5.0 * 2.0 + 6.0 * 1.0 = 16.0