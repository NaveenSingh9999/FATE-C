"""Core tensor abstraction for FATE-C."""

import numpy as np
from typing import Optional, Union, Tuple, List, Any


class Tensor:
    """Lightweight tensor wrapper with autograd support."""
    
    def __init__(self, data, requires_grad=False, dtype=None):
        if isinstance(data, Tensor):
            self.data = data.data
            self.requires_grad = requires_grad or data.requires_grad
        else:
            self.data = np.asarray(data, dtype=dtype)
            self.requires_grad = requires_grad
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def numpy(self):
        """Return the underlying NumPy array."""
        return self.data
    
    def __repr__(self):
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad})"
    
    def __add__(self, other):
        from .ops import add
        return add(self, other)
    
    def __mul__(self, other):
        from .ops import mul
        return mul(self, other)
    
    def __matmul__(self, other):
        from .ops import matmul
        return matmul(self, other)


def tensor(data, requires_grad=False, dtype=None):
    """Create a new tensor."""
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def zeros(*shape, requires_grad=False, dtype=None):
    """Create a tensor filled with zeros."""
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def randn(*shape, requires_grad=False, dtype=None):
    """Create a tensor with random normal values."""
    return Tensor(np.random.randn(*shape).astype(dtype or np.float32), requires_grad=requires_grad)
