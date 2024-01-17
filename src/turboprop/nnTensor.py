import turboprop as tp
from abc import ABC, abstractmethod
import random
import numpy as np


class nn(ABC):
    @abstractmethod
    def parameters(self):
        pass

    # reset gradients to zeros (to do after each gradient descent step)
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros(p.shape)


# Dense Layer (Linear + optional ReLU)
class Dense(nn):
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


class Sequential(nn):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


X = np.random.randn(28 * 28, 500)

model = Sequential([
    Dense(28 * 28, 128),
    Dense(128, 64),
    Dense(64, 10, relu=False)
    ])
print(model.parameters())
print(model(X))
