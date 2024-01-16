# import numpy as np

# Build an auto backprop package from Numpy, following and extending on the tinygrad tutorial by A. Karpathy 


class Scalar:
    def __init__(self, value, _prev=(), _op=None):
        self.value = value
        self.grad = 0

        # for auto backprop
        self._prev = _prev
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f'Scalar({self.value})'

    def __add__(self, other):
        res = Scalar(self.value + other.value, _prev=(self, other), _op='+')

        def _backward():
            self.grad += res.grad
            other.grad += res.grad
        res._backward = _backward
        return res

    def __mul__(self, other):
        res = Scalar(self.value * other.value, _prev=(self, other), _op='*')

        def _backward():
            self.grad += res.grad * other.value
            other.grad += res.grad * self.value
        res._backward = _backward
        return res

    def backward(self):
        self.grad = 1
        # dfs build of topological order of nodes
        visited = set()
        topo = []

        def build_topo(node):
            if node in visited:
                return
            visited.add(node)
            for prev_node in node._prev:
                build_topo(prev_node)
            topo.append(node)
        build_topo(self)

        # backprop of grad to all _prev variables in reverse topological order
        print(topo)
        for node in reversed(topo):
            node._backward()

''''
a = Scalar(5.0)
print(a)

b = Scalar(-2.0)
print(b)

c = a + b
print(c)

d = Scalar(4)
print(d)

e = c * d
e.backward()
print(a.grad)
print(b.grad)
print(c.grad)
print(d.grad)
print(e.grad)
'''