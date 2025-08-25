"""Built-in activation functions."""

import numpy as np
from typing import Union, Callable, Optional
from .registry import neuron


@neuron
def relu(x):
    """ReLU activation function."""
    from ..core.tensor import Tensor
    if not isinstance(x, Tensor):
        x = Tensor(x)
    result = np.maximum(0, x.data)
    return Tensor(result, requires_grad=x.requires_grad)


@neuron
def sigmoid(x):
    """Sigmoid activation function."""
    from ..core.tensor import Tensor
    if not isinstance(x, Tensor):
        x = Tensor(x)
    result = 1 / (1 + np.exp(-x.data))
    return Tensor(result, requires_grad=x.requires_grad)


@neuron
def tanh(x):
    """Tanh activation function."""
    from ..core.tensor import Tensor
    if not isinstance(x, Tensor):
        x = Tensor(x)
    result = np.tanh(x.data)
    return Tensor(result, requires_grad=x.requires_grad)


@neuron
def gelu(x):
    """GELU activation function."""
    from ..core.tensor import Tensor
    if not isinstance(x, Tensor):
        x = Tensor(x)
    result = 0.5 * x.data * (1 + np.tanh(np.sqrt(2/np.pi) * (x.data + 0.044715 * x.data**3)))
    return Tensor(result, requires_grad=x.requires_grad)


@neuron
def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function."""
    from ..core.tensor import Tensor
    if not isinstance(x, Tensor):
        x = Tensor(x)
    result = np.where(x.data > 0, x.data, alpha * x.data)
    return Tensor(result, requires_grad=x.requires_grad)


@neuron
def elu(x, alpha=1.0):
    """ELU activation function."""
    from ..core.tensor import Tensor
    if not isinstance(x, Tensor):
        x = Tensor(x)
    result = np.where(x.data > 0, x.data, alpha * (np.exp(x.data) - 1))
    return Tensor(result, requires_grad=x.requires_grad)


@neuron
def softmax(x, axis=-1):
    """Softmax activation function."""
    from ..core.tensor import Tensor
    if not isinstance(x, Tensor):
        x = Tensor(x)
    # Subtract max for numerical stability
    x_shifted = x.data - np.max(x.data, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    result = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    return Tensor(result, requires_grad=x.requires_grad)


@neuron
def swish(x):
    """Swish activation function."""
    from ..core.tensor import Tensor
    if not isinstance(x, Tensor):
        x = Tensor(x)
    sig = sigmoid(x)
    result = x.data * sig.data
    return Tensor(result, requires_grad=x.requires_grad)


# Import intelligent neurons
from .intelligent import SmartReLU, AdaptiveSigmoid


def smart_relu(x):
    """Smart ReLU with intelligence for 97%+ targeting."""
    from ..core.tensor import Tensor
    if not isinstance(x, Tensor):
        x = Tensor(x)
    neuron = SmartReLU()
    return neuron(x)


def adaptive_sigmoid(x):
    """Adaptive Sigmoid with intelligence for 97%+ targeting."""
    from ..core.tensor import Tensor
    if not isinstance(x, Tensor):
        x = Tensor(x)
    neuron = AdaptiveSigmoid()
    return neuron(x)


def intelligent_relu(x):
    """Alias for smart_relu."""
    return smart_relu(x)


def smart_sigmoid(x):
    """Alias for adaptive_sigmoid."""
    return adaptive_sigmoid(x)


def linear_activation(x):
    """Linear activation (identity function)."""
    from ..core.tensor import Tensor
    if not isinstance(x, Tensor):
        x = Tensor(x)
    return x


# Activation function registry
ACTIVATION_FUNCTIONS = {
    'relu': relu,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'gelu': gelu,
    'leaky_relu': leaky_relu,
    'elu': elu,
    'softmax': softmax,
    'swish': swish,
    'smart_relu': smart_relu,
    'adaptive_sigmoid': adaptive_sigmoid,
    'intelligent_relu': intelligent_relu,
    'smart_sigmoid': smart_sigmoid,
    'linear': linear_activation,
    'none': linear_activation,
    None: linear_activation,
}


def get_activation(activation: Union[str, Callable, None]) -> Optional[Callable]:
    """
    Get activation function by name or return the function if already callable.
    
    Args:
        activation: Activation function name (str), callable, or None
        
    Returns:
        Activation function or None
        
    Raises:
        ValueError: If activation name is not recognized
    """
    if activation is None:
        return None
    
    if callable(activation):
        return activation
    
    if isinstance(activation, str):
        activation_lower = activation.lower()
        if activation_lower in ACTIVATION_FUNCTIONS:
            return ACTIVATION_FUNCTIONS[activation_lower]
        else:
            available = list(ACTIVATION_FUNCTIONS.keys())
            raise ValueError(f"Unknown activation function '{activation}'. Available: {available}")
    
    raise TypeError(f"Activation must be string, callable, or None, got {type(activation)}")


def list_activations():
    """List all available activation functions."""
    return list(ACTIVATION_FUNCTIONS.keys())


def register_activation(name: str, func: Callable):
    """Register a custom activation function."""
    if not callable(func):
        raise TypeError("Activation function must be callable")
    
    ACTIVATION_FUNCTIONS[name.lower()] = func
