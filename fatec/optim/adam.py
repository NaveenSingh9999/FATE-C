"""Adam optimizer."""

import numpy as np


class Adam:
    """Adam optimizer."""
    
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.m = {}
        self.v = {}
        self.t = 0
    
    def step(self, parameters, gradients):
        """Update parameters using gradients."""
        self.t += 1
        
        for name, param in parameters.items():
            if name not in gradients:
                continue
            
            grad = gradients[name]
            
            if name not in self.m:
                self.m[name] = np.zeros_like(param.data)
                self.v[name] = np.zeros_like(param.data)
            
            # Simple Adam update
            self.m[name] = 0.9 * self.m[name] + 0.1 * grad
            self.v[name] = 0.999 * self.v[name] + 0.001 * grad**2
            
            param.data -= self.learning_rate * self.m[name] / (np.sqrt(self.v[name]) + 1e-8)
