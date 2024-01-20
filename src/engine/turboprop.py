"""The Scalar and Tensor classes for turbograd are wrappers around numpy scalar
and 2D arrays. The objects keep a reference of the (_prev) variables used in
their computation to form a computational graph. Their _backward method is used
to backpropagate the gradient through the graph.
"""
import numpy as np


class Scalar:
    """Wrapper around numpy float32 scalar. At the moment, Scalar is only used
    as the endpoint of the computational graph (cross entropy loss). Its 
    backward method is called to initiate the gradient backpropagation.
    """
    def __init__(self, value, _prev=(), label='n/a'):
        assert value.ndim == 0 and value.dtype == np.float32
        self.value = value
        self.grad = 0
        self.label = label  # for debug

        # For backprop
        self._prev = _prev
        self._backward = None

    def __repr__(self):
        return f'TgScalar({self.value}), grad={self.grad}'

    def backward(self):
        self.grad = 1

        # DFS build of topological order of nodes
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

        # print('Built reverse topo order is:')
        # print([node.label for node in reversed(topo)])

        # Backprop of grad through the compute graph in rev. topological order
        for node in reversed(topo):
            if node._backward is not None:
                node._backward()


class Tensor:
    """Wrapper around numpy 2D arrays. Used for the matrix operations of the
    forward propagation and for building the corresponding computational
    graph. Each matrix operation contains a _backward method to backpropagate
    the gradient
    """
    def __init__(self, array: np.ndarray, _prev=(), label='n/a'):
        assert isinstance(array, np.ndarray) and array.ndim == 2
        self.array = array
        self.grad = np.zeros(array.shape)
        self.label = label

        # For backprop
        self._prev = _prev
        self._backward = None

    def __repr__(self):
        return f'TgTensor(shape={self.array.shape}, dtype={self.array.dtype})'

    def __matmul__(self, other):
        assert (isinstance(other, np.ndarray) and other.ndim == 2) \
            or isinstance(other, Tensor)
        if isinstance(other, Tensor):
            other_arr = other.array
            _prev = (self, other)
        else:
            other_arr = other
            _prev = (self,)
        assert other_arr.shape[0] == self.array.shape[1]  # for matmul
        label = other.label if isinstance(other, Tensor) else 'X'
        res = Tensor(self.array @ other_arr, _prev=_prev,
                     label=self.label + '@' + label)

        def _backward():
            self.grad += res.grad @ other_arr.T
            if isinstance(other, Tensor):
                other.grad += self.array.T @ res.grad
        res._backward = _backward
        return res

    def __add__(self, other):
        assert isinstance(other, Tensor) and other.array.ndim == 2
        assert other.array.shape == self.array.shape \
            or other.array.shape == (self.array.shape[0], 1)
        res = Tensor(self.array + other.array, _prev=(self, other),
                     label=self.label + '+' + other.label)

        def _backward():
            self.grad += res.grad
            if other.array.shape[1] == 1:
                other.grad += res.grad.sum(axis=1, keepdims=True)
            else:
                other.grad += res.grad
        res._backward = _backward
        return res

    def relu(self):
        res = Tensor(np.maximum(0, self.array), _prev=(self,),
                     label=f'Relu({self.label})')

        def _backward():
            self.grad += res.grad
            self.grad[self.array <= 0] = 0
        res._backward = _backward
        return res
