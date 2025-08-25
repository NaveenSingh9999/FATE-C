"""Enhanced Dense layer for FATE-C Production."""

import numpy as np
from typing import Optional, Union, Callable
from ..core.tensor import Tensor, zeros
from ..core import ops
from ..neurons.activations import get_activation
from .base import Layer


class Dense(Layer):
    """Enhanced Dense (fully connected) layer with production features."""
    
    def __init__(self, 
                 units: int,
                 activation: Optional[Union[str, Callable]] = None,
                 use_bias: bool = True,
                 kernel_initializer: str = 'xavier_uniform',
                 bias_initializer: str = 'zeros',
                 kernel_constraint: Optional[Callable] = None,
                 bias_constraint: Optional[Callable] = None,
                 dropout_rate: float = 0.0,
                 name: Optional[str] = None,
                 **kwargs):
        """
        Initialize Dense layer.
        
        Args:
            units: Number of output units
            activation: Activation function name or callable
            use_bias: Whether to use bias
            kernel_initializer: Weight initialization method
            bias_initializer: Bias initialization method
            kernel_constraint: Constraint function for weights
            bias_constraint: Constraint function for bias
            dropout_rate: Dropout rate (0.0 = no dropout)
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        
        # Validate parameters
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        
        # Configuration
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.dropout_rate = dropout_rate
        
        # Store config
        self.config.update(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            dropout_rate=dropout_rate
        )
        
        # Parameters (initialized in build)
        self.weight = None
        self.bias = None
        
        # Get activation function
        self._activation_fn = get_activation(activation) if activation else None
    
    def _build_layer(self, input_shape):
        """Build layer parameters based on input shape."""
        if len(input_shape) < 2:
            raise ValueError(f"Dense layer requires at least 2D input, got {len(input_shape)}D")
        
        input_dim = input_shape[-1]
        
        # Add weight parameter
        self.weight = self.add_parameter(
            'weight', 
            (input_dim, self.units), 
            self.kernel_initializer,
            self.kernel_constraint
        )
        
        # Add bias parameter if needed
        if self.use_bias:
            self.bias = self.add_parameter(
                'bias',
                (self.units,),
                self.bias_initializer,
                self.bias_constraint
            )
    
    def forward(self, x: Tensor) -> Tensor:
        """Enhanced forward pass with validation and optimizations."""
        # Input validation
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        # Shape validation
        if x.ndim < 2:
            raise ValueError(f"Dense layer requires at least 2D input, got {x.ndim}D")
        
        if x.shape[-1] != self.weight.data.shape[0]:
            raise ValueError(f"Input feature dimension {x.shape[-1]} doesn't match weight dimension {self.weight.data.shape[0]}")
        
        # Store input shape for potential reshaping
        original_shape = x.shape
        
        # Reshape to 2D if needed (batch processing)
        if x.ndim > 2:
            batch_size = np.prod(original_shape[:-1])
            x = x.reshape(batch_size, original_shape[-1])
        
        # Linear transformation: x @ W
        output = ops.matmul(x, self.weight.data)
        
        # Add bias if enabled
        if self.use_bias:
            output = ops.add(output, self.bias.data)
        
        # Apply activation function
        if self._activation_fn:
            output = self._activation_fn(output)
        
        # Apply dropout during training
        if self.dropout_rate > 0.0 and self._training:
            output = self._apply_dropout(output)
        
        # Reshape back to original batch structure
        if len(original_shape) > 2:
            new_shape = original_shape[:-1] + (self.units,)
            output = output.reshape(new_shape)
        
        return output
    
    def _apply_dropout(self, x: Tensor) -> Tensor:
        """Apply dropout during training."""
        if self.dropout_rate == 0.0 or not self._training:
            return x
        
        # Generate dropout mask
        keep_prob = 1.0 - self.dropout_rate
        mask = np.random.binomial(1, keep_prob, x.shape) / keep_prob
        
        # Apply mask
        return ops.mul(x, Tensor(mask.astype(self.dtype), requires_grad=False))
    
    def get_output_shape(self, input_shape):
        """Calculate output shape given input shape."""
        return input_shape[:-1] + (self.units,)
    
    def compute_output_shape(self, input_shape):
        """Alias for get_output_shape for compatibility."""
        return self.get_output_shape(input_shape)
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            units=self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            dropout_rate=self.dropout_rate
        )
        return config
    
    def count_params(self):
        """Count total number of parameters."""
        params = self.weight.data.size
        if self.use_bias:
            params += self.bias.data.size
        return params
    
    def get_weights(self):
        """Get layer weights as numpy arrays."""
        weights = [self.weight.data.data]
        if self.use_bias:
            weights.append(self.bias.data.data)
        return weights
    
    def set_weights(self, weights):
        """Set layer weights from numpy arrays."""
        if len(weights) != (2 if self.use_bias else 1):
            raise ValueError(f"Expected {2 if self.use_bias else 1} weight arrays, got {len(weights)}")
        
        # Set weight
        if weights[0].shape != self.weight.data.shape:
            raise ValueError(f"Weight shape mismatch: expected {self.weight.data.shape}, got {weights[0].shape}")
        self.weight.data.data = weights[0].astype(self.dtype)
        
        # Set bias if applicable
        if self.use_bias:
            if weights[1].shape != self.bias.data.shape:
                raise ValueError(f"Bias shape mismatch: expected {self.bias.data.shape}, got {weights[1].shape}")
            self.bias.data.data = weights[1].astype(self.dtype)
    
    def __repr__(self):
        return (f"Dense(units={self.units}, activation={self.activation}, "
                f"use_bias={self.use_bias}, dropout_rate={self.dropout_rate}, "
                f"name={self.name})")
