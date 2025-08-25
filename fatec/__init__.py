"""
FATE-C: Universal Neural Network Designer & Compiler
"""

__version__ = "0.1.0"

# Core imports for flat API
from .core.tensor import Tensor
from .core.autograd import grad, no_grad
from .core import ops

# Neuron system
from .neurons.registry import neuron, get_neuron
from .neurons.activations import *

# Layers
from .layers.dense import Dense
from .layers.dropout import Dropout
from .layers.base import Layer

# Models
from .models.sequential import Sequential
from .models.base import BaseModel

# Training
from .training.trainer import Trainer

# Optimizers
from .optim.sgd import SGD
from .optim.adam import Adam

# High-level API functions
def seq(layers):
    """Create a sequential model from a list of layers."""
    return Sequential(layers)

# Export common symbols
__all__ = [
    'Tensor', 'grad', 'no_grad', 'ops',
    'neuron', 'get_neuron',
    'Dense', 'Dropout', 'Layer',
    'Sequential', 'BaseModel', 'seq',
    'Trainer', 'SGD', 'Adam'
]
