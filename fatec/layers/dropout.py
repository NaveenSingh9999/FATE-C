"""Dropout layer."""

import numpy as np
from ..core.tensor import Tensor
from .base import Layer


class Dropout(Layer):
    """Dropout layer for regularization."""
    
    def __init__(self, rate=0.5, name=None):
        super().__init__(name)
        self.rate = rate
        self.training = True
    
    def forward(self, x):
        """Forward pass."""
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        if self.training and self.rate > 0:
            keep_prob = 1.0 - self.rate
            mask = np.random.binomial(1, keep_prob, x.shape) / keep_prob
            return Tensor(x.data * mask, requires_grad=x.requires_grad)
        
        return x
