"""Built-in activation functions."""

import numpy as np
from .registry import neuron


@neuron
def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


@neuron
def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


@neuron
def tanh(x):
    """Tanh activation function."""
    return np.tanh(x)


@neuron
def gelu(x):
    """GELU activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
