"""Core operations for FATE-C."""

import numpy as np
from .autograd import Function


class Add(Function):
    """Addition operation."""
    
    def forward(self, a, b):
        return a + b
    
    def backward(self, grad_output):
        return [grad_output, grad_output]


class Mul(Function):
    """Multiplication operation."""
    
    def forward(self, a, b):
        return a * b
    
    def backward(self, grad_output):
        grad_a = grad_output * self.inputs[1].data
        grad_b = grad_output * self.inputs[0].data
        return [grad_a, grad_b]


class MatMul(Function):
    """Matrix multiplication operation."""
    
    def forward(self, a, b):
        return np.matmul(a, b)
    
    def backward(self, grad_output):
        grad_a = np.matmul(grad_output, self.inputs[1].data.T)
        grad_b = np.matmul(self.inputs[0].data.T, grad_output)
        return [grad_a, grad_b]


class ReLU(Function):
    """ReLU activation function."""
    
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return [grad_output * (self.inputs[0].data > 0)]


def add(a, b):
    """Add two tensors."""
    result = Add()(a, b)
    if hasattr(a, 'requires_grad') or hasattr(b, 'requires_grad'):
        from .tensor import Tensor
        return Tensor(result.data, requires_grad=result.requires_grad)
    return result


def mul(a, b):
    """Multiply two tensors."""
    result = Mul()(a, b)
    if hasattr(a, 'requires_grad') or hasattr(b, 'requires_grad'):
        from .tensor import Tensor
        return Tensor(result.data, requires_grad=result.requires_grad)
    return result


def matmul(a, b):
    """Matrix multiply two tensors."""
    result = MatMul()(a, b)
    if hasattr(a, 'requires_grad') or hasattr(b, 'requires_grad'):
        from .tensor import Tensor
        return Tensor(result.data, requires_grad=result.requires_grad)
    return result


def relu(x):
    """Apply ReLU activation."""
    result = ReLU()(x)
    if hasattr(x, 'requires_grad'):
        from .tensor import Tensor
        return Tensor(result.data, requires_grad=result.requires_grad)
    return result
