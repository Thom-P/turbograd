"""Classes for model construction loosely following the Pytorch nomenclature:
Dense initializes a fully connected linear layer with optional ReLU activation.
Sequential stacks a series of linear/ReLU layers.
CrossEntropyLoss computes the eponyme loss function from the output logits as
well as the associated logits gradients.
"""
from abc import ABC, abstractmethod

import numpy as np

import turbograd.wrappers as tw


class Module(ABC):
    @abstractmethod
    def parameters(self):
        pass

    # Reset gradients to zeros (to do after each gradient descent step)
    def zero_grad(self):
        for p in self.parameters():
            p.grad.fill(0)
            # p.grad = np.zeros(p.array.shape)


class Dense(Module):
    """Linear layer with optional ReLU activation, the weights are initialized
    from a random uniform ditribution normalized with the number of inputs.
    The biases are initialized to 0.
    """
    def __init__(self, n_in, n_out, relu=True, label='n/a'):
        self.label = label
        stdv = 1. / np.sqrt(n_in)
        weights = np.random.uniform(-stdv, stdv,
                                    size=(n_out, n_in)).astype(np.float32)
        self.weights = tw.Tensor(weights, label=self.label + '_W')
        self.biases = tw.Tensor(np.zeros((n_out, 1)).astype(np.float32),
                                label=self.label + '_b')
        self.relu = relu

    def __call__(self, A):
        Z = self.weights @ A + self.biases
        return Z.relu() if self.relu else Z

    def parameters(self):
        return [self.weights, self.biases]

    def __repr__(self):
        layer_type = 'Linear + ReLU' if self.relu else 'Linear'
        return (f'Dense {layer_type} layer {self.label},'
                f'size = {self.weights.shape}')


class Sequential(Module):
    """Sequence of layers that forms a model"""
    def __init__(self, layers, label='myModel'):
        self.layers = layers
        self.label = label

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# CrossEntropy gradient calculation:
# https://www.michaelpiseno.com/blog/2021/softmax-gradient/
class CrossEntropyLoss():
    """Cross-entropy loss calculation function from the output logits of a
    model and the index of the target class (averaged over the batch examples).
    The cross entropy loss corresponds to the negative log of the softmax
    output (i.e. probabibily) of the target class. The logits gradients are
    also computed for the backward pass.
    """
    def __call__(self, Z, y):
        assert isinstance(y, np.ndarray) and y.shape == (1, Z.array.shape[1])
        assert y.dtype == int  # expect indices
        batch_size = Z.array.shape[1]
        max_vals = Z.array.max(axis=0, keepdims=True)
        exp_Z = np.exp(Z.array - max_vals)  # - max_vals to avoid exp overflow
        softmax_denom = exp_Z.sum(axis=0, keepdims=True)
        Z_select = Z.array[y, np.arange(batch_size)] - max_vals
        loss = (-Z_select + np.log(softmax_denom)).mean(axis=1).squeeze()
        res = tw.Scalar(loss, _prev=(Z,), label='crossLoss')

        def _backward():
            softmax = exp_Z / softmax_denom
            softmax[y, np.arange(batch_size)] -= 1
            softmax /= batch_size  # because mean used in fwd calculation.
            Z.grad += softmax
        res._backward = _backward
        return res
