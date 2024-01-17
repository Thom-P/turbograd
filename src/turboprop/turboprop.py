# import numpy as np
# Build an auto backprop package from Numpy,
# following (class Scalar) and extending (class Tensor) on the tinygrad tutorial by A. Karpathy
import math
import numpy as np


class Scalar:
    def __init__(self, value, _prev=()):
        self.value = value
        self.grad = 0

        # for auto backprop
        self._prev = _prev
        self._backward = lambda: None

    def __repr__(self):
        return f'Scalar({self.value}), grad={self.grad}'

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        res = Scalar(self.value + other.value, _prev=(self, other))

        def _backward():
            self.grad += res.grad
            other.grad += res.grad
        res._backward = _backward
        return res

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        res = Scalar(self.value * other.value, _prev=(self, other))

        def _backward():
            self.grad += res.grad * other.value
            other.grad += res.grad * self.value
        res._backward = _backward
        return res

    def __neg__(self):
        return self * -1

    def __pow__(self, pow):
        res = Scalar(self.value ** pow, _prev=(self,))

        def _backward():
            self.grad += res.grad * pow * (self.value ** (pow - 1))
        res._backward = _backward
        return res

    def __truediv__(self, other):
        return self * (other ** -1)

    def log(self):
        res = Scalar(math.log(self.value), _prev=(self,))

        def _backward():
            self.grad += res.grad / self.value
        res._backward = _backward
        return res

    def relu(self):
        res = Scalar(max(0, self.value), _prev=(self,))

        def _backward():
            self.grad += res.grad * (1 if self.value >= 0 else 0)
        res._backward = _backward
        return res

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

        # backprop of grad to all _prev variables in reverse topological order
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


class Tensor:
    def __init__(self, array: np.ndarray, _prev=()):
        assert isinstance(array, np.ndarray) and array.ndim == 2
        self.array = array
        self.grad = np.zeros(array.shape)

        # for auto backprop
        self._prev = _prev
        self._backward = lambda: None

    def __repr__(self):
        return f'Tensor(shape={self.array.shape}, dtype={self.array.dtype})'

    def __matmul__(self, other):
        assert (isinstance(other, np.ndarray) and other.ndim == 2) \
            or isinstance(other, Tensor)
        if isinstance(other, Tensor):
            other_arr = other.array
            _prev = (self, other)
        else:
            other_arr = other
            _prev = (self,)
        assert other_arr.shape[0] == self.array.shape[1]  # ok for matmul
        res = Tensor(self.array @ other_arr, _prev=_prev)

        def _backward():
            self.grad += 1 / other_arr.shape[1] * \
                res.grad @ other_arr.T
            if isinstance(other, Tensor):
                other.grad += 1 / self.array.shape[0] * \
                    self.array.T @ res.grad  # need to dbl check that
        res._backward = _backward
        return res

    def __add__(self, other):
        assert isinstance(other, Tensor) and other.array.ndim == 2
        assert other.array.shape == self.array.shape \
            or other.array.shape == (self.array.shape[0], 1)
        res = Tensor(self.array + other.array, _prev=(self, other))

        def _backward():
            self.grad += res.grad
            if other.array.shape[1] == 1:
                other.grad += 1 / self.array.shape[1] \
                    * res.grad.sum(axis=1, keepdims=True)
            else:
                other.grad += res.grad
        res._backward = _backward
        return res

    def relu(self):
        self.array = np.maximum(0, self.array) # in place ok?
        return self

    def __mul__(self, other):
        assert isinstance(other, float)  # limited to float atm
        self.array = self.array * other  # no need to track op i think(?)
        return self

    def __truediv__(self, other):
        assert isinstance(other, Tensor)
        res = Tensor(self.array / other.array, _prev=(self, other))

        def _backward():
            self.grad = res.grad / other.array
            other.grad = res.grad * -self.array / (other.array ** 2)
        res._backward = _backward
        return res

    def __neg__(self):
        self.array = -self.array  # idem
        return self

    def log(self):
        res = Tensor(np.log(self.array), _prev=(self,))

        def _backward():
            self.grad = res.grad / self.array
        res._backward = _backward
        return res

    def exp(self):
        res = Tensor(np.exp(self.array), _prev=(self,))

        def _backward():
            self.grad = res.grad * np.exp(self.array)
        res._backward = _backward
        return res
    


'''
A = np.random.rand(30, 40)
B = np.random.rand(40, 60)
B_bad = np.random.rand(39, 60) 
C1 = np.random.rand(30, 60)  
C2 = np.random.rand(30, 1)  
tA = Tensor(A)
tB = Tensor(B)
tBb = Tensor(B_bad)
tC1 = Tensor(C1)
tC2 = Tensor(C2)

tD = tA @ tB + tC2
print(tD)

expo = tA.exp()
expo = (-tA).exp()
logo = tB.log()
relo = tC1.relu()
divo = tC1 * 3. / tC2
'''