"""Dense layer."""

import numpy as np
from ..core.tensor import Tensor, zeros
from ..core import ops
from .base import Layer


class Dense(Layer):
    """Dense (fully connected) layer."""
    
    def __init__(self, units, activation=None, use_bias=True, name=None):
        super().__init__(name)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.weight = None
        self.bias = None
    
    def build(self, input_shape):
        """Build layer parameters."""
        input_dim = input_shape[-1]
        
        # Xavier initialization
        limit = np.sqrt(6.0 / (input_dim + self.units))
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (input_dim, self.units)),
            requires_grad=True
        )
        
        if self.use_bias:
            self.bias = zeros(self.units, requires_grad=True)
        
        self.parameters['weight'] = self.weight
        if self.use_bias:
            self.parameters['bias'] = self.bias
        
        super().build(input_shape)
    
    def forward(self, x):
        """Forward pass."""
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        output = ops.matmul(x, self.weight)
        
        if self.use_bias:
            output = ops.add(output, self.bias)
        
        if self.activation == 'relu':
            output = ops.relu(output)
        
        return output
