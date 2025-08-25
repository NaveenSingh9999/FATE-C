"""Enhanced Base layer class for FATE-C Production."""

from abc import ABC, abstractmethod
import numpy as np
import uuid
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union


class LayerConfig:
    """Configuration class for layer parameters."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        
    def get(self, key, default=None):
        return self.config.get(key, default)
        
    def update(self, **kwargs):
        self.config.update(kwargs)
        
    def to_dict(self):
        return self.config.copy()


class Parameter:
    """Enhanced parameter class with metadata."""
    
    def __init__(self, data, name=None, requires_grad=True, constraint=None):
        from ..core.tensor import Tensor
        
        if isinstance(data, Tensor):
            self.data = data
        else:
            self.data = Tensor(data, requires_grad=requires_grad)
            
        self.name = name
        self.constraint = constraint
        self._updates = 0
        self._creation_time = uuid.uuid4().hex[:8]
        
    def apply_constraint(self):
        """Apply constraint to parameter if defined."""
        if self.constraint:
            self.data = self.constraint(self.data)
            
    def __repr__(self):
        return f"Parameter(name={self.name}, shape={self.data.shape}, updates={self._updates})"


class Layer(ABC):
    """Enhanced base class for all layers with production features."""
    
    def __init__(self, name=None, trainable=True, dtype=np.float32, **kwargs):
        # Core properties
        self.name = name or f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self.built = False
        self.trainable = trainable
        self.dtype = dtype
        
        # Parameters and state
        self._parameters = {}
        self._buffers = {}
        self._modules = {}
        
        # Configuration
        self.config = LayerConfig(**kwargs)
        
        # Metadata
        self._input_shape = None
        self._output_shape = None
        self._forward_calls = 0
        self._training = True
        
        # Layer hierarchy
        self._parent = None
        self._children = []
        
    def __repr__(self):
        param_count = sum(p.data.size for p in self._parameters.values())
        return f"{self.__class__.__name__}(name={self.name}, params={param_count}, trainable={self.trainable})"
    
    def add_parameter(self, name: str, shape: Tuple[int, ...], 
                     initializer: str = 'xavier_uniform', constraint=None) -> Parameter:
        """Add a parameter to the layer."""
        if name in self._parameters:
            raise ValueError(f"Parameter '{name}' already exists")
            
        # Initialize parameter
        data = self._initialize_parameter(shape, initializer)
        param = Parameter(data, name=f"{self.name}.{name}", constraint=constraint)
        
        self._parameters[name] = param
        return param
    
    def add_buffer(self, name: str, data: np.ndarray):
        """Add a non-trainable buffer to the layer."""
        from ..core.tensor import Tensor
        
        if name in self._buffers:
            raise ValueError(f"Buffer '{name}' already exists")
            
        if not isinstance(data, Tensor):
            data = Tensor(data, requires_grad=False)
            
        self._buffers[name] = data
    
    def _initialize_parameter(self, shape: Tuple[int, ...], initializer: str) -> np.ndarray:
        """Initialize parameter with given shape and initializer."""
        if initializer == 'zeros':
            return np.zeros(shape, dtype=self.dtype)
        elif initializer == 'ones':
            return np.ones(shape, dtype=self.dtype)
        elif initializer == 'xavier_uniform':
            limit = np.sqrt(6.0 / (shape[0] + shape[-1]))
            return np.random.uniform(-limit, limit, shape).astype(self.dtype)
        elif initializer == 'xavier_normal':
            std = np.sqrt(2.0 / (shape[0] + shape[-1]))
            return np.random.normal(0, std, shape).astype(self.dtype)
        elif initializer == 'he_uniform':
            limit = np.sqrt(6.0 / shape[0])
            return np.random.uniform(-limit, limit, shape).astype(self.dtype)
        elif initializer == 'he_normal':
            std = np.sqrt(2.0 / shape[0])
            return np.random.normal(0, std, shape).astype(self.dtype)
        elif initializer == 'normal':
            return np.random.normal(0, 0.01, shape).astype(self.dtype)
        elif initializer == 'uniform':
            return np.random.uniform(-0.1, 0.1, shape).astype(self.dtype)
        else:
            raise ValueError(f"Unknown initializer: {initializer}")
    
    def build(self, input_shape: Tuple[int, ...]):
        """Build the layer with given input shape."""
        if self.built:
            warnings.warn(f"Layer {self.name} is already built", RuntimeWarning)
            return
            
        self._input_shape = tuple(input_shape)
        self._build_layer(input_shape)
        self.built = True
        
        # Validate build
        self._validate_build()
    
    def _build_layer(self, input_shape: Tuple[int, ...]):
        """Layer-specific build logic - override in subclasses."""
        pass
    
    def _validate_build(self):
        """Validate layer after building."""
        if not self.built:
            raise RuntimeError(f"Layer {self.name} build validation failed")
            
        # Check parameter shapes
        for name, param in self._parameters.items():
            if param.data.data.size == 0:
                raise ValueError(f"Parameter {name} has zero size")
    
    @abstractmethod
    def forward(self, x):
        """Forward pass through the layer - must be implemented by subclasses."""
        pass
    
    def __call__(self, x):
        """Enhanced call method with validation and metadata tracking."""
        from ..core.tensor import Tensor
        
        # Input validation
        if not isinstance(x, Tensor):
            if hasattr(x, 'data'):
                x = Tensor(x.data)
            else:
                x = Tensor(x)
        
        # Auto-build if needed
        if not self.built:
            if hasattr(x, 'shape'):
                input_shape = x.shape
            else:
                input_shape = x.data.shape
            self.build(input_shape)
        
        # Shape validation
        if self._input_shape and x.shape[1:] != self._input_shape[1:]:
            warnings.warn(f"Input shape {x.shape} doesn't match expected {self._input_shape}", RuntimeWarning)
        
        # Forward pass
        try:
            self._forward_calls += 1
            output = self.forward(x)
            
            # Output validation
            if not isinstance(output, Tensor):
                raise RuntimeError(f"Layer {self.name} must return a Tensor")
                
            self._output_shape = output.shape
            
            # Apply constraints to parameters
            for param in self._parameters.values():
                param.apply_constraint()
            
            return output
            
        except Exception as e:
            raise RuntimeError(f"Forward pass failed in layer {self.name}: {e}")
    
    def train(self, mode: bool = True):
        """Set layer to training mode."""
        self._training = mode
        for child in self._children:
            child.train(mode)
        return self
    
    def eval(self):
        """Set layer to evaluation mode."""
        return self.train(False)
    
    def get_parameters(self) -> Dict[str, Parameter]:
        """Get all trainable parameters."""
        if not self.trainable:
            return {}
        return self._parameters.copy()
    
    def get_buffers(self) -> Dict[str, Any]:
        """Get all buffers."""
        return self._buffers.copy()
    
    def named_parameters(self, prefix='') -> List[Tuple[str, Parameter]]:
        """Get named parameters with optional prefix."""
        result = []
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            result.append((full_name, param))
        
        for name, child in self._modules.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            result.extend(child.named_parameters(child_prefix))
        
        return result
    
    def parameters(self) -> List[Parameter]:
        """Get all parameters as a list."""
        return [param for _, param in self.named_parameters()]
    
    def zero_grad(self):
        """Zero gradients for all parameters."""
        for param in self._parameters.values():
            if param.data.grad is not None:
                param.data.zero_grad()
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = {
            'name': self.name,
            'trainable': self.trainable,
            'dtype': str(self.dtype),
            'built': self.built,
        }
        config.update(self.config.to_dict())
        return config
    
    def get_stats(self) -> Dict[str, Any]:
        """Get layer statistics."""
        param_count = sum(p.data.size for p in self._parameters.values())
        buffer_count = sum(b.size for b in self._buffers.values())
        
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'parameters': param_count,
            'buffers': buffer_count,
            'forward_calls': self._forward_calls,
            'input_shape': self._input_shape,
            'output_shape': self._output_shape,
            'trainable': self.trainable,
            'training': self._training,
        }
    
    def summary(self) -> str:
        """Get layer summary string."""
        stats = self.get_stats()
        lines = [
            f"Layer: {stats['name']} ({stats['type']})",
            f"  Parameters: {stats['parameters']:,}",
            f"  Input Shape: {stats['input_shape']}",
            f"  Output Shape: {stats['output_shape']}",
            f"  Trainable: {stats['trainable']}",
            f"  Forward Calls: {stats['forward_calls']:,}"
        ]
        return '\n'.join(lines)
    
    def reset_parameters(self):
        """Reset all parameters to initial values."""
        for name, param in self._parameters.items():
            # Re-initialize with same shape
            new_data = self._initialize_parameter(param.data.shape, 'xavier_uniform')
            param.data.data = new_data
            param._updates = 0
    
    def freeze(self):
        """Freeze layer parameters (stop training)."""
        self.trainable = False
        for param in self._parameters.values():
            param.data.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze layer parameters (resume training)."""
        self.trainable = True
        for param in self._parameters.values():
            param.data.requires_grad = True
