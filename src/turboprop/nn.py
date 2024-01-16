import turboprop as tp
from abc import ABC, abstractmethod
import random


class nn(ABC):
    @abstractmethod
    def parameters(self):
        pass

    # reset gradients to zeros (to do after each gradient descent step)
    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0


class Neuron(nn):
    def __init__(self, n_input, relu=True):
        self.W = [tp.Scalar(random.gauss(mu=0, sigma=2 / n_input))
                  for _ in range(n_input)]
        self.b = tp.Scalar(0)
        self.relu = relu

    def __call__(self, x):
        z = self.b + sum(wi * xi for wi, xi in zip(self.W, x))
        return z.relu() if self.relu else z

    def parameters(self):
        return self.W + [self.b]

    def __repr__(self):
        neur_type = "Relu" if self.relu else "Linear"
        return f'{neur_type} neuron with {len(self.W)} weights'


# Dense Layer (Linear + optional ReLU)
class Dense(nn):
    def __init__(self, n_in, n_out, relu=True):
        self.neurons = [Neuron(n_in, relu=relu) for _ in range(n_out)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class Sequential(nn):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# n = Neuron(10, relu=True)
# x = list(range(1, 11))
# a = n(x)
# print(n.parameters())

# layer = Dense(4, 2)
# print(layer.parameters())
# print(layer(range(1, 11)))

# layers = [Dense(20, 10), Dense(10, 5), Dense(5, 2, relu=False)]
x = [random.random() for _ in range(20)]
model = Sequential([
    Dense(20, 10),
    Dense(10, 5),
    Dense(5, 2, relu=False)
    ])
print(model.parameters())
print(model(x))
