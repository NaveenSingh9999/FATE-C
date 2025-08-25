"""
FATE-C Production Tensor System

Enhanced tensor implementation with:
- Advanced memory management
- Performance optimizations
- Device support (CPU/GPU ready)
- Comprehensive operations
- Error handling and validation
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Any, Callable
import warnings


class Tensor:
    """
    Production-ready tensor with advanced features.
    
    Features:
    - Memory-efficient operations
    - Device abstraction
    - Broadcasting support
    - Type safety
    - Performance monitoring
    """
    
    def __init__(self, data, requires_grad=False, dtype=None, device='cpu', name=None):
        # Input validation
        if data is None:
            raise ValueError("Tensor data cannot be None")
        
        if isinstance(data, Tensor):
            self.data = data.data.copy()  # Deep copy for safety
            self.requires_grad = requires_grad or data.requires_grad
            self.device = device or data.device
            # dtype is determined by self.data.dtype via property
        else:
            try:
                self.data = np.asarray(data, dtype=dtype)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot convert data to tensor: {e}")
            
            self.requires_grad = requires_grad
            self.device = device
            # dtype is determined by self.data.dtype via property
        
        # Metadata
        self.name = name
        self.grad = None
        self._version = 0
        self._creation_context = self._get_creation_context()
        
        # Performance tracking
        self._access_count = 0
        self._last_modified = None
        
        # Validation
        self._validate_tensor()
    
    def _validate_tensor(self):
        """Validate tensor state and properties."""
        if self.data.size == 0:
            warnings.warn("Creating empty tensor", UserWarning)
        
        if self.data.size > 1e9:  # 1GB limit
            warnings.warn("Large tensor detected, consider chunking", UserWarning)
        
        if np.any(np.isnan(self.data)):
            warnings.warn("Tensor contains NaN values", UserWarning)
        
        if np.any(np.isinf(self.data)):
            warnings.warn("Tensor contains infinite values", UserWarning)
    
    def _get_creation_context(self):
        """Get context information for debugging."""
        import traceback
        stack = traceback.extract_stack()
        return {
            'file': stack[-3].filename if len(stack) > 2 else 'unknown',
            'line': stack[-3].lineno if len(stack) > 2 else 0,
            'function': stack[-3].name if len(stack) > 2 else 'unknown'
        }
    
    @property
    def shape(self):
        """Get tensor shape."""
        self._access_count += 1
        return self.data.shape
    
    @property
    def size(self):
        """Get total number of elements."""
        return self.data.size
    
    @property
    def ndim(self):
        """Get number of dimensions."""
        return self.data.ndim
    
    @property
    def dtype(self):
        """Get data type."""
        return self.data.dtype
    
    @property
    def memory_usage(self):
        """Get memory usage in bytes."""
        return self.data.nbytes
    
    def numpy(self):
        """Return the underlying NumPy array."""
        self._access_count += 1
        return self.data
    
    def clone(self):
        """Create a deep copy of the tensor."""
        return Tensor(
            self.data.copy(), 
            requires_grad=self.requires_grad,
            dtype=self.dtype,
            device=self.device,
            name=f"{self.name}_clone" if self.name else None
        )
    
    def detach(self):
        """Detach from computation graph."""
        result = self.clone()
        result.requires_grad = False
        result.grad = None
        return result
    
    def to(self, dtype=None, device=None):
        """Convert tensor to different dtype or device."""
        new_dtype = dtype or self.dtype
        new_device = device or self.device
        
        if new_dtype == self.dtype and new_device == self.device:
            return self
        
        new_data = self.data.astype(new_dtype) if new_dtype != self.dtype else self.data
        
        return Tensor(
            new_data,
            requires_grad=self.requires_grad,
            dtype=new_dtype,
            device=new_device,
            name=self.name
        )
    
    def reshape(self, *shape):
        """Reshape tensor."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        
        try:
            new_data = self.data.reshape(shape)
            return Tensor(
                new_data,
                requires_grad=self.requires_grad,
                dtype=self.dtype,
                device=self.device,
                name=f"{self.name}_reshaped" if self.name else None
            )
        except ValueError as e:
            raise ValueError(f"Cannot reshape tensor from {self.shape} to {shape}: {e}")
    
    def flatten(self):
        """Flatten tensor to 1D."""
        return self.reshape(-1)
    
    def transpose(self, dim0=None, dim1=None):
        """Transpose tensor."""
        if dim0 is None and dim1 is None:
            # Full transpose
            new_data = self.data.T
        else:
            # Specific dimensions
            if dim0 is None or dim1 is None:
                raise ValueError("Both dimensions must be specified for partial transpose")
            
            axes = list(range(self.ndim))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            new_data = self.data.transpose(axes)
        
        return Tensor(
            new_data,
            requires_grad=self.requires_grad,
            dtype=self.dtype,
            device=self.device,
            name=f"{self.name}_T" if self.name else None
        )
    
    def sum(self, axis=None, keepdims=False):
        """Sum along axis."""
        return Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            device=self.device
        )
    
    def mean(self, axis=None, keepdims=False):
        """Mean along axis."""
        return Tensor(
            np.mean(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            device=self.device
        )
    
    def max(self, axis=None, keepdims=False):
        """Maximum along axis."""
        return Tensor(
            np.max(self.data, axis=axis, keepdims=keepdims),
            requires_grad=False,  # Max is not differentiable everywhere
            device=self.device
        )
    
    def min(self, axis=None, keepdims=False):
        """Minimum along axis."""
        return Tensor(
            np.min(self.data, axis=axis, keepdims=keepdims),
            requires_grad=False,  # Min is not differentiable everywhere
            device=self.device
        )
    
    def argmax(self, axis=None):
        """Indices of maximum values."""
        return np.argmax(self.data, axis=axis)
    
    def argmin(self, axis=None):
        """Indices of minimum values."""
        return np.argmin(self.data, axis=axis)
    
    def abs(self):
        """Absolute value."""
        return Tensor(
            np.abs(self.data),
            requires_grad=self.requires_grad,
            device=self.device,
            name=f"abs({self.name})" if self.name else None
        )
    
    def sqrt(self):
        """Square root."""
        return Tensor(
            np.sqrt(self.data),
            requires_grad=self.requires_grad,
            device=self.device,
            name=f"sqrt({self.name})" if self.name else None
        )
    
    def exp(self):
        """Exponential."""
        return Tensor(
            np.exp(self.data),
            requires_grad=self.requires_grad,
            device=self.device,
            name=f"exp({self.name})" if self.name else None
        )
    
    def log(self):
        """Natural logarithm."""
        return Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            device=self.device,
            name=f"log({self.name})" if self.name else None
        )
    
    def sin(self):
        """Sine."""
        return Tensor(
            np.sin(self.data),
            requires_grad=self.requires_grad,
            device=self.device,
            name=f"sin({self.name})" if self.name else None
        )
    
    def cos(self):
        """Cosine."""
        return Tensor(
            np.cos(self.data),
            requires_grad=self.requires_grad,
            device=self.device,
            name=f"cos({self.name})" if self.name else None
        )
    
    def pow(self, exponent):
        """Power operation."""
        return Tensor(
            np.power(self.data, exponent),
            requires_grad=self.requires_grad,
            device=self.device,
            name=f"pow({self.name}, {exponent})" if self.name else None
        )
    
    def __getitem__(self, key):
        """Advanced indexing with error handling."""
        try:
            result_data = self.data[key]
            return Tensor(
                result_data,
                requires_grad=self.requires_grad,
                dtype=self.dtype,
                device=self.device,
                name=f"{self.name}[{key}]" if self.name else None
            )
        except (IndexError, TypeError) as e:
            raise IndexError(f"Invalid index {key} for tensor with shape {self.shape}: {e}")
    
    def __setitem__(self, key, value):
        """In-place assignment with validation."""
        try:
            if isinstance(value, Tensor):
                self.data[key] = value.data
            else:
                self.data[key] = value
            self._version += 1
        except (IndexError, ValueError, TypeError) as e:
            raise ValueError(f"Cannot assign value to index {key}: {e}")
    
    def __repr__(self):
        """Enhanced string representation."""
        shape_str = 'x'.join(map(str, self.shape))
        device_str = f", device='{self.device}'" if self.device != 'cpu' else ""
        grad_str = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        name_str = f", name='{self.name}'" if self.name else ""
        
        return f"Tensor(shape=[{shape_str}], dtype={self.dtype}{device_str}{grad_str}{name_str})"
    
    def __str__(self):
        """Human-readable string representation."""
        return f"Tensor({self.data!r})"
    
    def info(self):
        """Detailed tensor information."""
        return {
            'shape': self.shape,
            'dtype': str(self.dtype),
            'device': self.device,
            'requires_grad': self.requires_grad,
            'memory_usage': f"{self.memory_usage / 1024:.2f} KB",
            'access_count': self._access_count,
            'name': self.name,
            'creation_context': self._creation_context,
            'has_nan': bool(np.any(np.isnan(self.data))),
            'has_inf': bool(np.any(np.isinf(self.data))),
            'min_value': float(np.min(self.data)),
            'max_value': float(np.max(self.data)),
            'mean_value': float(np.mean(self.data)),
            'std_value': float(np.std(self.data))
        }
    
    
    # Arithmetic Operations with Broadcasting Support
    def _ensure_tensor(self, other):
        """Convert input to tensor with validation."""
        if not isinstance(other, Tensor):
            if isinstance(other, (int, float, complex)):
                other = Tensor([other], device=self.device, dtype=self.dtype)
            else:
                other = Tensor(other, device=self.device)
        return other
    
    def _check_broadcasting(self, other):
        """Check if tensors can be broadcasted."""
        try:
            np.broadcast_shapes(self.shape, other.shape)
            return True
        except ValueError:
            return False
    
    def __add__(self, other):
        """Enhanced addition with broadcasting."""
        other = self._ensure_tensor(other)
        
        if not self._check_broadcasting(other):
            raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")
        
        from .ops import add
        return add(self, other)
    
    def __radd__(self, other):
        """Reverse addition."""
        return self.__add__(other)
    
    def __sub__(self, other):
        """Subtraction."""
        other = self._ensure_tensor(other)
        
        if not self._check_broadcasting(other):
            raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")
        
        from .ops import sub
        return sub(self, other)
    
    def __rsub__(self, other):
        """Reverse subtraction."""
        other = self._ensure_tensor(other)
        from .ops import sub
        return sub(other, self)
    
    def __mul__(self, other):
        """Enhanced multiplication."""
        other = self._ensure_tensor(other)
        
        if not self._check_broadcasting(other):
            raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")
        
        from .ops import mul
        return mul(self, other)
    
    def __rmul__(self, other):
        """Reverse multiplication."""
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Division."""
        other = self._ensure_tensor(other)
        
        if not self._check_broadcasting(other):
            raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")
        
        # Check for division by zero
        if np.any(other.data == 0):
            warnings.warn("Division by zero detected", RuntimeWarning)
        
        from .ops import div
        return div(self, other)
    
    def __rtruediv__(self, other):
        """Reverse division."""
        other = self._ensure_tensor(other)
        from .ops import div
        return div(other, self)
    
    def __pow__(self, other):
        """Power operation."""
        if isinstance(other, (int, float)):
            return self.pow(other)
        else:
            other = self._ensure_tensor(other)
            return Tensor(
                np.power(self.data, other.data),
                requires_grad=self.requires_grad or other.requires_grad,
                device=self.device
            )
    
    def __matmul__(self, other):
        """Enhanced matrix multiplication."""
        other = self._ensure_tensor(other)
        
        # Validate shapes for matrix multiplication
        if self.ndim < 2 or other.ndim < 2:
            raise ValueError("Matrix multiplication requires at least 2D tensors")
        
        if self.shape[-1] != other.shape[-2]:
            raise ValueError(f"Cannot multiply matrices with shapes {self.shape} and {other.shape}")
        
        from .ops import matmul
        return matmul(self, other)
    
    def __neg__(self):
        """Negation."""
        return Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            device=self.device,
            name=f"-{self.name}" if self.name else None
        )
    
    def __abs__(self):
        """Absolute value."""
        return self.abs()
    
    # Comparison Operations
    def __eq__(self, other):
        """Element-wise equality."""
        other = self._ensure_tensor(other)
        return Tensor(self.data == other.data, device=self.device)
    
    def __ne__(self, other):
        """Element-wise inequality."""
        other = self._ensure_tensor(other)
        return Tensor(self.data != other.data, device=self.device)
    
    def __lt__(self, other):
        """Element-wise less than."""
        other = self._ensure_tensor(other)
        return Tensor(self.data < other.data, device=self.device)
    
    def __le__(self, other):
        """Element-wise less than or equal."""
        other = self._ensure_tensor(other)
        return Tensor(self.data <= other.data, device=self.device)
    
    def __gt__(self, other):
        """Element-wise greater than."""
        other = self._ensure_tensor(other)
        return Tensor(self.data > other.data, device=self.device)
    
    def __ge__(self, other):
        """Element-wise greater than or equal."""
        other = self._ensure_tensor(other)
        return Tensor(self.data >= other.data, device=self.device)


def tensor(data, requires_grad=False, dtype=None, device='cpu', name=None):
    """
    Create a new tensor with enhanced options.
    
    Args:
        data: Input data (array-like)
        requires_grad: Whether to track gradients
        dtype: Data type
        device: Device placement ('cpu', 'gpu')
        name: Optional name for debugging
    
    Returns:
        Tensor: New tensor instance
    """
    return Tensor(data, requires_grad=requires_grad, dtype=dtype, device=device, name=name)


def zeros(shape, dtype=np.float32, device='cpu', requires_grad=False, name=None):
    """Create tensor filled with zeros."""
    return Tensor(
        np.zeros(shape, dtype=dtype),
        requires_grad=requires_grad,
        device=device,
        name=name or 'zeros'
    )


def ones(shape, dtype=np.float32, device='cpu', requires_grad=False, name=None):
    """Create tensor filled with ones."""
    return Tensor(
        np.ones(shape, dtype=dtype),
        requires_grad=requires_grad,
        device=device,
        name=name or 'ones'
    )


def randn(*shape, dtype=np.float32, device='cpu', requires_grad=False, name=None):
    """Create tensor with random normal values."""
    return Tensor(
        np.random.randn(*shape).astype(dtype),
        requires_grad=requires_grad,
        device=device,
        name=name or 'randn'
    )


def rand(*shape, dtype=np.float32, device='cpu', requires_grad=False, name=None):
    """Create tensor with random uniform values."""
    return Tensor(
        np.random.rand(*shape).astype(dtype),
        requires_grad=requires_grad,
        device=device,
        name=name or 'rand'
    )


def eye(n, dtype=np.float32, device='cpu', requires_grad=False, name=None):
    """Create identity matrix."""
    return Tensor(
        np.eye(n, dtype=dtype),
        requires_grad=requires_grad,
        device=device,
        name=name or 'eye'
    )


def arange(start, stop=None, step=1, dtype=np.float32, device='cpu', requires_grad=False, name=None):
    """Create tensor with evenly spaced values."""
    if stop is None:
        stop = start
        start = 0
    
    return Tensor(
        np.arange(start, stop, step, dtype=dtype),
        requires_grad=requires_grad,
        device=device,
        name=name or 'arange'
    )


def linspace(start, stop, num=50, dtype=np.float32, device='cpu', requires_grad=False, name=None):
    """Create tensor with linearly spaced values."""
    return Tensor(
        np.linspace(start, stop, num, dtype=dtype),
        requires_grad=requires_grad,
        device=device,
        name=name or 'linspace'
    )


# Tensor utilities
def cat(tensors, axis=0):
    """Concatenate tensors along axis."""
    if not tensors:
        raise ValueError("Cannot concatenate empty list of tensors")
    
    arrays = [t.data for t in tensors]
    result_data = np.concatenate(arrays, axis=axis)
    
    # Determine if any tensor requires grad
    requires_grad = any(t.requires_grad for t in tensors)
    
    return Tensor(
        result_data,
        requires_grad=requires_grad,
        device=tensors[0].device,
        name='cat'
    )


def stack(tensors, axis=0):
    """Stack tensors along new axis."""
    if not tensors:
        raise ValueError("Cannot stack empty list of tensors")
    
    arrays = [t.data for t in tensors]
    result_data = np.stack(arrays, axis=axis)
    
    requires_grad = any(t.requires_grad for t in tensors)
    
    return Tensor(
        result_data,
        requires_grad=requires_grad,
        device=tensors[0].device,
        name='stack'
    )


def zeros(*shape, requires_grad=False, dtype=None):
    """Create a tensor filled with zeros."""
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def randn(*shape, requires_grad=False, dtype=None):
    """Create a tensor with random normal values."""
    return Tensor(np.random.randn(*shape).astype(dtype or np.float32), requires_grad=requires_grad)
