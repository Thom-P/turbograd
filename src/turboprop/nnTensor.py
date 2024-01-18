import turboprop as tp
from abc import ABC, abstractmethod
import random
import numpy as np


class Module(ABC):
    @abstractmethod
    def parameters(self):
        pass

    # reset gradients to zeros (to do after each gradient descent step)
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros(p.shape)


# Dense Layer (Linear + optional ReLU)
class Dense(Module):
    def __init__(self, n_in, n_out, relu=True):
        sigma, mu = 2, 0
        self.weights = tp.Tensor(sigma * np.random.randn(n_out, n_in) + mu)
        self.biases = tp.Tensor(np.zeros((n_out, 1)))
        self.relu = relu

    def __call__(self, A):
        Z = self.weights @ A + self.biases
        return Z.relu() if self.relu else Z

    def parameters(self):
        return [self.weights, self.biases]
    
    #def __repr__(self):
    #    neur_type = "Relu" if self.relu else "Linear"
    #    return f'{neur_type} neuron with {len(self.W)} weights'


# gradient calc: https://www.michaelpiseno.com/blog/2021/softmax-gradient/
class CrossEntropyLoss():
    # def __init__(self)

    def __call__(self, Z, y):
        assert isinstance(y, np.ndarray) and y.shape == (1, Z.array.shape[1])
        assert y.dtype == int  # expect indices
        batch_size = Z.array.shape[1]
        exp_Z = np.exp(Z.array)
        softmax_denom = exp_Z.sum(axis=0, keepdims=True)
        Z_select = Z.array[y, np.arange(batch_size)]
        loss = (-Z_select + np.log(softmax_denom)).mean(axis=1)  # mean instead of sum (default on pytorch)
        res = tp.Scalar(loss, _prev=(Z,))
        # now backward...

        def _backward():
            softmax = exp_Z / softmax_denom
            Z.grad = softmax / batch_size  # because mean used in fwd
            Z.grad[y, np.arange(batch_size)] -= 1 / batch_size  # because mean used in fwd
        res._backward = _backward
        return res


class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

'''
X = np.random.randn(28 * 28, 500)
y = np.random.randint(0, 10, size=(1, 500))

model = Sequential([
    Dense(28 * 28, 128),
    Dense(128, 64),
    Dense(64, 10, relu=False)
    ])
#print(model.parameters())
#print(model(X))
Z = model(X)
loss_fn = CategoricalCrossEntropy()
loss = loss_fn(Z, y)
loss.backward()
'''