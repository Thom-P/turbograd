import turboprop as tp
#from abc import ABC, abstractmethod
import random


#class nn(ABC):
class nn:
    #@abstractmethod
    def parameters(self):
        pass

    # reset gradients to zeros (to do after each gradient descent step)
    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0


class Neuron(nn):
    def __init__(self, n_input, relu=True):
        self.W = [tp.Scalar(random.gauss(mu=0, sigma=2 / n_input)) for _ in range(n_input)]
        self.b = tp.Scalar(0)
        self.relu = relu

        def __call__(self, x):
            z = self.b + sum(wi * xi for wi, xi in zip(self.W, x))
            return z.relu() if self.relu else z

        def parameters(self):
            return self.W + [self.b]

        def __repr__(self):
            return f'{"Relu" if self.relu else "Linear"} Neuron with {len(self.W)} weights'

n = Neuron(10)
print(n)