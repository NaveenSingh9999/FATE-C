"""SGD optimizer."""

import numpy as np


class SGD:
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def step(self, parameters, gradients):
        """Update parameters using gradients."""
        for name, param in parameters.items():
            if name in gradients:
                grad = gradients[name]
                param.data -= self.learning_rate * grad
