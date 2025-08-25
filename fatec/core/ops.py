"""
FATE-C Production Operations

Enhanced tensor operations with:
- Comprehensive error handling
- Performance optimizations
- Broadcasting support
- Memory efficiency
- Numerical stability
"""

import numpy as np
import warnings
from .autograd import Function
from typing import List, Union, Optional


class Add(Function):
    """Production-ready addition operation with broadcasting."""
    
    def forward(self, a, b):
        try:
            # Ensure broadcasting compatibility
            result = np.add(a, b)
            
            # Check for overflow
            if np.any(np.isinf(result)) and not (np.any(np.isinf(a)) or np.any(np.isinf(b))):
                warnings.warn("Addition overflow detected", RuntimeWarning)
            
            return result
        except ValueError as e:
            raise ValueError(f"Addition failed: {e}. Shapes: {a.shape} + {b.shape}")
    
    def backward(self, grad_output):
        # Handle broadcasting in gradients
        grad_a = grad_output
        grad_b = grad_output
        
        # Sum out added dims for proper gradient shape
        while grad_a.ndim > self.inputs[0].data.ndim:
            grad_a = grad_a.sum(axis=0)
        while grad_b.ndim > self.inputs[1].data.ndim:
            grad_b = grad_b.sum(axis=0)
        
        # Sum over broadcasted dimensions
        for i in range(grad_a.ndim):
            if self.inputs[0].data.shape[i] == 1 and grad_a.shape[i] > 1:
                grad_a = grad_a.sum(axis=i, keepdims=True)
        
        for i in range(grad_b.ndim):
            if self.inputs[1].data.shape[i] == 1 and grad_b.shape[i] > 1:
                grad_b = grad_b.sum(axis=i, keepdims=True)
        
        return [grad_a, grad_b]


class Sub(Function):
    """Subtraction operation with broadcasting."""
    
    def forward(self, a, b):
        try:
            result = np.subtract(a, b)
            
            if np.any(np.isinf(result)) and not (np.any(np.isinf(a)) or np.any(np.isinf(b))):
                warnings.warn("Subtraction overflow detected", RuntimeWarning)
            
            return result
        except ValueError as e:
            raise ValueError(f"Subtraction failed: {e}. Shapes: {a.shape} - {b.shape}")
    
    def backward(self, grad_output):
        grad_a = grad_output
        grad_b = -grad_output
        
        # Handle broadcasting
        while grad_a.ndim > self.inputs[0].data.ndim:
            grad_a = grad_a.sum(axis=0)
        while grad_b.ndim > self.inputs[1].data.ndim:
            grad_b = grad_b.sum(axis=0)
        
        for i in range(grad_a.ndim):
            if self.inputs[0].data.shape[i] == 1 and grad_a.shape[i] > 1:
                grad_a = grad_a.sum(axis=i, keepdims=True)
        
        for i in range(grad_b.ndim):
            if self.inputs[1].data.shape[i] == 1 and grad_b.shape[i] > 1:
                grad_b = grad_b.sum(axis=i, keepdims=True)
        
        return [grad_a, grad_b]


class Mul(Function):
    """Enhanced multiplication operation."""
    
    def forward(self, a, b):
        try:
            result = np.multiply(a, b)
            
            # Check for overflow
            if np.any(np.isinf(result)) and not (np.any(np.isinf(a)) or np.any(np.isinf(b))):
                warnings.warn("Multiplication overflow detected", RuntimeWarning)
            
            # Check for underflow
            if np.any(result == 0) and not (np.any(a == 0) or np.any(b == 0)):
                warnings.warn("Multiplication underflow detected", RuntimeWarning)
            
            return result
        except ValueError as e:
            raise ValueError(f"Multiplication failed: {e}. Shapes: {a.shape} * {b.shape}")
    
    def backward(self, grad_output):
        grad_a = grad_output * self.inputs[1].data
        grad_b = grad_output * self.inputs[0].data
        
        # Handle broadcasting
        while grad_a.ndim > self.inputs[0].data.ndim:
            grad_a = grad_a.sum(axis=0)
        while grad_b.ndim > self.inputs[1].data.ndim:
            grad_b = grad_b.sum(axis=0)
        
        for i in range(grad_a.ndim):
            if self.inputs[0].data.shape[i] == 1 and grad_a.shape[i] > 1:
                grad_a = grad_a.sum(axis=i, keepdims=True)
        
        for i in range(grad_b.ndim):
            if self.inputs[1].data.shape[i] == 1 and grad_b.shape[i] > 1:
                grad_b = grad_b.sum(axis=i, keepdims=True)
        
        return [grad_a, grad_b]


class Div(Function):
    """Division operation with numerical stability."""
    
    def forward(self, a, b):
        try:
            # Check for division by zero
            if np.any(b == 0):
                # Replace zeros with small epsilon for numerical stability
                b_safe = np.where(b == 0, 1e-8, b)
                warnings.warn("Division by zero, using epsilon=1e-8", RuntimeWarning)
                result = np.divide(a, b_safe)
            else:
                result = np.divide(a, b)
            
            # Check for overflow
            if np.any(np.isinf(result)) and not np.any(np.isinf(a)):
                warnings.warn("Division overflow detected", RuntimeWarning)
            
            return result
        except ValueError as e:
            raise ValueError(f"Division failed: {e}. Shapes: {a.shape} / {b.shape}")
    
    def backward(self, grad_output):
        b_data = self.inputs[1].data
        
        # Gradient w.r.t. a: grad_output / b
        grad_a = grad_output / b_data
        
        # Gradient w.r.t. b: -grad_output * a / (b^2)
        grad_b = -grad_output * self.inputs[0].data / (b_data ** 2)
        
        # Handle broadcasting
        while grad_a.ndim > self.inputs[0].data.ndim:
            grad_a = grad_a.sum(axis=0)
        while grad_b.ndim > self.inputs[1].data.ndim:
            grad_b = grad_b.sum(axis=0)
        
        for i in range(grad_a.ndim):
            if self.inputs[0].data.shape[i] == 1 and grad_a.shape[i] > 1:
                grad_a = grad_a.sum(axis=i, keepdims=True)
        
        for i in range(grad_b.ndim):
            if self.inputs[1].data.shape[i] == 1 and grad_b.shape[i] > 1:
                grad_b = grad_b.sum(axis=i, keepdims=True)
        
        return [grad_a, grad_b]


class MatMul(Function):
    """Enhanced matrix multiplication with validation."""
    
    def forward(self, a, b):
        try:
            # Validate dimensions
            if a.ndim < 2 or b.ndim < 2:
                raise ValueError(f"MatMul requires at least 2D tensors, got {a.ndim}D and {b.ndim}D")
            
            if a.shape[-1] != b.shape[-2]:
                raise ValueError(f"Cannot multiply matrices: {a.shape} @ {b.shape}")
            
            result = np.matmul(a, b)
            
            # Check for numerical issues
            if np.any(np.isnan(result)):
                warnings.warn("MatMul produced NaN values", RuntimeWarning)
            
            if np.any(np.isinf(result)):
                warnings.warn("MatMul produced infinite values", RuntimeWarning)
            
            return result
        except Exception as e:
            raise ValueError(f"Matrix multiplication failed: {e}")
    
    def backward(self, grad_output):
        a_data = self.inputs[0].data
        b_data = self.inputs[1].data
        
        # Gradient w.r.t. a: grad_output @ b.T
        grad_a = np.matmul(grad_output, np.swapaxes(b_data, -2, -1))
        
        # Gradient w.r.t. b: a.T @ grad_output
        grad_b = np.matmul(np.swapaxes(a_data, -2, -1), grad_output)
        
        return [grad_a, grad_b]


class Exp(Function):
    """Exponential operation with overflow protection."""
    
    def forward(self, x):
        # Clip to prevent overflow
        x_clipped = np.clip(x, -700, 700)  # Safe range for exp
        if not np.array_equal(x, x_clipped):
            warnings.warn("Input clipped to prevent exp overflow", RuntimeWarning)
        
        result = np.exp(x_clipped)
        self.output = result  # Store for backward pass
        return result
    
    def backward(self, grad_output):
        return [grad_output * self.output]


class Log(Function):
    """Logarithm operation with numerical stability."""
    
    def forward(self, x):
        # Ensure positive input
        x_safe = np.maximum(x, 1e-8)
        if not np.array_equal(x, x_safe):
            warnings.warn("Negative values clamped for log", RuntimeWarning)
        
        result = np.log(x_safe)
        self.input_safe = x_safe  # Store for backward pass
        return result
    
    def backward(self, grad_output):
        return [grad_output / self.input_safe]


class Sin(Function):
    """Sine operation."""
    
    def forward(self, x):
        result = np.sin(x)
        self.input = x  # Store for backward pass
        return result
    
    def backward(self, grad_output):
        return [grad_output * np.cos(self.input)]


class Cos(Function):
    """Cosine operation."""
    
    def forward(self, x):
        result = np.cos(x)
        self.input = x  # Store for backward pass
        return result
    
    def backward(self, grad_output):
        return [-grad_output * np.sin(self.input)]


class Pow(Function):
    """Power operation."""
    
    def forward(self, x, exponent):
        try:
            result = np.power(x, exponent)
            
            if np.any(np.isnan(result)) and not np.any(np.isnan(x)):
                warnings.warn("Power operation produced NaN", RuntimeWarning)
            
            self.x = x
            self.exponent = exponent
            return result
        except Exception as e:
            raise ValueError(f"Power operation failed: {e}")
    
    def backward(self, grad_output):
        # d/dx (x^n) = n * x^(n-1)
        grad_x = grad_output * self.exponent * np.power(self.x, self.exponent - 1)
        return [grad_x]


class Sum(Function):
    """Sum operation with axis support."""
    
    def forward(self, x, axis=None, keepdims=False):
        self.input_shape = x.shape
        self.axis = axis
        self.keepdims = keepdims
        
        return np.sum(x, axis=axis, keepdims=keepdims)
    
    def backward(self, grad_output):
        # Expand gradient to match input shape
        if self.axis is not None and not self.keepdims:
            # Add back reduced dimensions
            if isinstance(self.axis, int):
                grad_output = np.expand_dims(grad_output, self.axis)
            else:
                for ax in sorted(self.axis):
                    grad_output = np.expand_dims(grad_output, ax)
        
        # Broadcast to input shape
        grad_input = np.broadcast_to(grad_output, self.input_shape)
        return [grad_input]


class Mean(Function):
    """Mean operation with axis support."""
    
    def forward(self, x, axis=None, keepdims=False):
        self.input_shape = x.shape
        self.axis = axis
        self.keepdims = keepdims
        
        # Calculate number of elements being averaged
        if axis is None:
            self.n_elements = x.size
        else:
            if isinstance(axis, int):
                self.n_elements = x.shape[axis]
            else:
                self.n_elements = np.prod([x.shape[ax] for ax in axis])
        
        return np.mean(x, axis=axis, keepdims=keepdims)
    
    def backward(self, grad_output):
        # Gradient of mean is 1/n for each element
        grad_output = grad_output / self.n_elements
        
        # Expand gradient to match input shape
        if self.axis is not None and not self.keepdims:
            if isinstance(self.axis, int):
                grad_output = np.expand_dims(grad_output, self.axis)
            else:
                for ax in sorted(self.axis):
                    grad_output = np.expand_dims(grad_output, ax)
        
        grad_input = np.broadcast_to(grad_output, self.input_shape)
        return [grad_input]


# Convenience functions for operations
def add(a, b):
    """Add two tensors."""
    from .tensor import Tensor
    op = Add()
    result_data = op.forward(a.data, b.data)
    result = Tensor(result_data, requires_grad=a.requires_grad or b.requires_grad)
    
    if result.requires_grad:
        op.inputs = [a, b]
        result._backward_fn = op
    
    return result


def sub(a, b):
    """Subtract two tensors."""
    from .tensor import Tensor
    op = Sub()
    result_data = op.forward(a.data, b.data)
    result = Tensor(result_data, requires_grad=a.requires_grad or b.requires_grad)
    
    if result.requires_grad:
        op.inputs = [a, b]
        result._backward_fn = op
    
    return result


def mul(a, b):
    """Multiply two tensors."""
    from .tensor import Tensor
    op = Mul()
    result_data = op.forward(a.data, b.data)
    result = Tensor(result_data, requires_grad=a.requires_grad or b.requires_grad)
    
    if result.requires_grad:
        op.inputs = [a, b]
        result._backward_fn = op
    
    return result


def div(a, b):
    """Divide two tensors."""
    from .tensor import Tensor
    op = Div()
    result_data = op.forward(a.data, b.data)
    result = Tensor(result_data, requires_grad=a.requires_grad or b.requires_grad)
    
    if result.requires_grad:
        op.inputs = [a, b]
        result._backward_fn = op
    
    return result


def matmul(a, b):
    """Matrix multiply two tensors."""
    from .tensor import Tensor
    op = MatMul()
    result_data = op.forward(a.data, b.data)
    result = Tensor(result_data, requires_grad=a.requires_grad or b.requires_grad)
    
    if result.requires_grad:
        op.inputs = [a, b]
        result._backward_fn = op
    
    return result


def exp(x):
    """Exponential of tensor."""
    from .tensor import Tensor
    op = Exp()
    result_data = op.forward(x.data)
    result = Tensor(result_data, requires_grad=x.requires_grad)
    
    if result.requires_grad:
        op.inputs = [x]
        result._backward_fn = op
    
    return result


def log(x):
    """Natural logarithm of tensor."""
    from .tensor import Tensor
    op = Log()
    result_data = op.forward(x.data)
    result = Tensor(result_data, requires_grad=x.requires_grad)
    
    if result.requires_grad:
        op.inputs = [x]
        result._backward_fn = op
    
    return result


def sin(x):
    """Sine of tensor."""
    from .tensor import Tensor
    op = Sin()
    result_data = op.forward(x.data)
    result = Tensor(result_data, requires_grad=x.requires_grad)
    
    if result.requires_grad:
        op.inputs = [x]
        result._backward_fn = op
    
    return result


def cos(x):
    """Cosine of tensor."""
    from .tensor import Tensor
    op = Cos()
    result_data = op.forward(x.data)
    result = Tensor(result_data, requires_grad=x.requires_grad)
    
    if result.requires_grad:
        op.inputs = [x]
        result._backward_fn = op
    
    return result


def pow(x, exponent):
    """Power of tensor."""
    from .tensor import Tensor
    op = Pow()
    result_data = op.forward(x.data, exponent)
    result = Tensor(result_data, requires_grad=x.requires_grad)
    
    if result.requires_grad:
        op.inputs = [x]
        result._backward_fn = op
    
    return result
    
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
