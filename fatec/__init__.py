"""
FATE-C: Universal Neural Network Designer & Compiler
Enhanced with intelligent neurons and diverse data support.
"""

__version__ = "0.2.0"

# Core imports for flat API
from .core.tensor import Tensor
from .core.autograd import grad, no_grad
from .core import ops

# Neuron system (enhanced)
from .neurons.registry import neuron, get_neuron
from .neurons.activations import *
from .neurons.intelligent import SmartReLU, AdaptiveSigmoid, IntelligentNeuron

# Layers
from .layers.dense import Dense
from .layers.dropout import Dropout
from .layers.base import Layer

# Models
from .models.sequential import Sequential
from .models.base import BaseModel

"""
FATE-C: Universal Neural Network Designer & Compiler
Enhanced with intelligent neurons and diverse data support.
"""

__version__ = "0.2.0"

# Core imports for flat API
from .core.tensor import Tensor
from .core.autograd import grad, no_grad
from .core import ops

# Neuron system (enhanced)
from .neurons.registry import neuron, get_neuron
from .neurons.activations import *
from .neurons.intelligent import SmartReLU, AdaptiveSigmoid, IntelligentNeuron

# Layers
from .layers.dense import Dense
from .layers.dropout import Dropout
from .layers.base import Layer

# Models
from .models.sequential import Sequential
from .models.base import BaseModel

# Training (enhanced)
from .training.trainer import Trainer
from .training.enhanced import EnhancedTrainer

# Universal training function
from .train import train, evaluate, train_numpy, train_pytorch, train_tensorflow

# Build System - Universal API
from .build import build, seq, Model, Neuron
from .build import Dense as DenseBuilder, Dropout as DropoutBuilder

# Data loading
from .data.loaders import CSVLoader, TXTLoader

# Optimizers
from .optim.sgd import SGD
from .optim.adam import Adam

# Export common symbols
__all__ = [
    'Tensor', 'grad', 'no_grad', 'ops',
    'neuron', 'get_neuron',
    'Dense', 'Dropout', 'Layer',
    'Sequential', 'BaseModel', 'seq', 'build', 'Model', 'Neuron',
    'Trainer', 'SGD', 'Adam', 'train', 'evaluate'
]
