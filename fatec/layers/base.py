"""Base layer class."""

from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    """Base class for all layers."""
    
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.built = False
        self.trainable = True
        self.parameters = {}
    
    def build(self, input_shape):
        """Build the layer with given input shape."""
        self.built = True
    
    @abstractmethod
    def forward(self, x):
        """Forward pass through the layer."""
        pass
    
    def __call__(self, x):
        """Call the layer on input x."""
        if not self.built:
            if hasattr(x, 'shape'):
                input_shape = x.shape
            else:
                input_shape = np.array(x).shape
            self.build(input_shape)
        
        return self.forward(x)
    
    def get_parameters(self):
        """Get all trainable parameters."""
        return self.parameters if self.trainable else {}
