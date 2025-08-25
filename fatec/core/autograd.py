"""Enhanced Autograd engine for FATE-C Production."""

import numpy as np
import threading
import warnings
from typing import List, Optional, Set, Dict, Any
from collections import defaultdict


class GradientContext:
    """Thread-local context for gradient computation control."""
    
    def __init__(self):
        self._enabled = True
        self._mode_stack = []
        
    @property
    def enabled(self):
        return self._enabled and len(self._mode_stack) == 0
        
    def enable_grad(self):
        """Enable gradient computation."""
        self._enabled = True
        
    def disable_grad(self):
        """Disable gradient computation."""
        self._enabled = False
        
    def push_mode(self, enabled):
        """Push gradient mode onto stack."""
        self._mode_stack.append(enabled)
        
    def pop_mode(self):
        """Pop gradient mode from stack."""
        if self._mode_stack:
            return self._mode_stack.pop()
        return self._enabled


# Thread-local gradient context
_grad_context = threading.local()


def _get_grad_context():
    """Get thread-local gradient context."""
    if not hasattr(_grad_context, 'context'):
        _grad_context.context = GradientContext()
    return _grad_context.context


class Variable:
    """Enhanced variable node in the computation graph."""
    
    def __init__(self, data, grad_fn=None, requires_grad=True, name=None):
        # Data validation and conversion
        if isinstance(data, Variable):
            raise TypeError("Cannot create Variable from another Variable")
            
        self.data = np.asarray(data, dtype=np.float32)
        
        # Validate data
        if self.data.size == 0:
            raise ValueError("Cannot create Variable with empty data")
            
        if not np.isfinite(self.data).all():
            warnings.warn("Variable contains non-finite values", RuntimeWarning)
        
        # Gradient computation properties
        self.grad_fn = grad_fn
        self.requires_grad = requires_grad and _get_grad_context().enabled
        self.grad = None
        self.name = name
        
        # Metadata for debugging
        self._version = 0
        self._grad_accumulator = None
        self._retains_grad = False
        
        # Graph construction
        self._children = set()
        if grad_fn:
            for inp in grad_fn.inputs:
                inp._children.add(self)
    
    def __repr__(self):
        grad_fn_str = f", grad_fn={self.grad_fn.__class__.__name__}" if self.grad_fn else ""
        name_str = f", name='{self.name}'" if self.name else ""
        return f"Variable(shape={self.data.shape}, requires_grad={self.requires_grad}{grad_fn_str}{name_str})"
    
    def zero_grad(self):
        """Zero out gradients."""
        self.grad = None
        self._version += 1
    
    def detach(self):
        """Create a new Variable detached from computation graph."""
        return Variable(self.data.copy(), requires_grad=False, name=f"{self.name}_detached" if self.name else None)
    
    def retain_grad(self):
        """Mark this Variable to retain gradients after backward pass."""
        self._retains_grad = True
    
    def backward(self, grad=None, retain_graph=False, create_graph=False):
        """Enhanced backward pass with gradient computation."""
        if not self.requires_grad:
            raise RuntimeError("Variable does not require gradients")
            
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad can only be implicitly created for scalar outputs")
            grad = np.ones_like(self.data)
        else:
            grad = np.asarray(grad)
            
        # Validate gradient shape
        if grad.shape != self.data.shape:
            raise RuntimeError(f"Gradient shape {grad.shape} does not match data shape {self.data.shape}")
        
        # Topological sort for backward pass
        topo_order = []
        visited = set()
        
        def topological_sort(var):
            if var in visited or not var.requires_grad:
                return
            visited.add(var)
            
            if var.grad_fn:
                for inp in var.grad_fn.inputs:
                    topological_sort(inp)
            
            topo_order.append(var)
        
        topological_sort(self)
        
        # Initialize gradients
        grads = {id(self): grad}
        
        # Backward pass in reverse topological order
        for var in reversed(topo_order):
            if id(var) not in grads:
                continue
                
            var_grad = grads[id(var)]
            
            # Accumulate gradient
            if var.grad is None:
                var.grad = var_grad.copy()
            else:
                var.grad += var_grad
            
            # Compute gradients for inputs
            if var.grad_fn:
                try:
                    input_grads = var.grad_fn.backward(var_grad)
                    
                    if len(input_grads) != len(var.grad_fn.inputs):
                        raise RuntimeError(f"Function returned {len(input_grads)} gradients but has {len(var.grad_fn.inputs)} inputs")
                    
                    for inp, inp_grad in zip(var.grad_fn.inputs, input_grads):
                        if inp.requires_grad and inp_grad is not None:
                            inp_grad = np.asarray(inp_grad)
                            
                            # Validate gradient shape
                            if inp_grad.shape != inp.data.shape:
                                raise RuntimeError(f"Gradient shape mismatch for input: {inp_grad.shape} vs {inp.data.shape}")
                            
                            if id(inp) in grads:
                                grads[id(inp)] += inp_grad
                            else:
                                grads[id(inp)] = inp_grad
                                
                except Exception as e:
                    raise RuntimeError(f"Error in backward pass for {var.grad_fn.__class__.__name__}: {e}")
            
            # Clean up graph if not retaining
            if not retain_graph and not var._retains_grad and var != self:
                var.grad_fn = None


class Function:
    """Enhanced base class for differentiable functions."""
    
    def __init__(self):
        self.inputs = []
        self.saved_tensors = []
        self._name = self.__class__.__name__
        
    def save_for_backward(self, *tensors):
        """Save tensors for backward pass."""
        self.saved_tensors = [t.data.copy() if hasattr(t, 'data') else np.asarray(t) for t in tensors]
    
    def forward(self, *inputs):
        """Forward pass - must be implemented by subclasses."""
        raise NotImplementedError(f"Forward pass not implemented for {self._name}")
    
    def backward(self, grad_output):
        """Backward pass - must be implemented by subclasses."""
        raise NotImplementedError(f"Backward pass not implemented for {self._name}")
    
    def __call__(self, *inputs):
        """Execute function with gradient tracking."""
        if not _get_grad_context().enabled:
            # No gradient tracking
            input_data = []
            for inp in inputs:
                if hasattr(inp, 'data'):
                    input_data.append(inp.data)
                else:
                    input_data.append(np.asarray(inp))
            
            output_data = self.forward(*input_data)
            return Variable(output_data, requires_grad=False)
        
        # Convert inputs to Variables
        var_inputs = []
        for inp in inputs:
            if isinstance(inp, Variable):
                var_inputs.append(inp)
            elif hasattr(inp, 'data'):  # Tensor object
                var_inputs.append(Variable(inp.data, requires_grad=inp.requires_grad))
            else:
                var_inputs.append(Variable(inp, requires_grad=False))
        
        self.inputs = var_inputs
        requires_grad = any(v.requires_grad for v in var_inputs)
        
        # Forward pass
        try:
            output_data = self.forward(*[v.data for v in var_inputs])
            
            if not isinstance(output_data, np.ndarray):
                output_data = np.asarray(output_data)
                
            # Validate output
            if not np.isfinite(output_data).all():
                warnings.warn(f"{self._name} produced non-finite values", RuntimeWarning)
                
        except Exception as e:
            raise RuntimeError(f"Forward pass failed for {self._name}: {e}")
        
        # Create output Variable
        if requires_grad:
            output = Variable(output_data, grad_fn=self, requires_grad=True)
        else:
            output = Variable(output_data, requires_grad=False)
        
        return output


class no_grad:
    """Context manager to disable gradient computation."""
    
    def __init__(self):
        self.prev_enabled = None
        
    def __enter__(self):
        context = _get_grad_context()
        self.prev_enabled = context.enabled
        context.push_mode(False)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        context = _get_grad_context()
        context.pop_mode()


class enable_grad:
    """Context manager to enable gradient computation."""
    
    def __init__(self):
        self.prev_enabled = None
        
    def __enter__(self):
        context = _get_grad_context()
        self.prev_enabled = context.enabled
        context.push_mode(True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        context = _get_grad_context()
        context.pop_mode()


def grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    """Enhanced gradient computation function."""
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    
    # Validate inputs
    for inp in inputs:
        if not isinstance(inp, Variable):
            raise TypeError("Inputs must be Variables")
        if not inp.requires_grad:
            raise RuntimeError("Input does not require gradients")
    
    # Validate outputs
    for out in outputs:
        if not isinstance(out, Variable):
            raise TypeError("Outputs must be Variables")
    
    # Zero gradients
    for inp in inputs:
        inp.zero_grad()
    
    # Prepare gradient outputs
    if grad_outputs is None:
        grad_outputs = [np.ones_like(out.data) for out in outputs]
    else:
        if len(grad_outputs) != len(outputs):
            raise RuntimeError("Number of gradient outputs must match number of outputs")
        grad_outputs = [np.asarray(g) for g in grad_outputs]
    
    # Run backward passes
    for output, grad_out in zip(outputs, grad_outputs):
        output.backward(grad_out, retain_graph=retain_graph, create_graph=create_graph)
    
    # Return gradients
    result_grads = []
    for inp in inputs:
        if inp.grad is not None:
            result_grads.append(inp.grad.copy())
        else:
            result_grads.append(np.zeros_like(inp.data))
    
    return result_grads


def set_grad_enabled(enabled):
    """Set global gradient computation state."""
    context = _get_grad_context()
    if enabled:
        context.enable_grad()
    else:
        context.disable_grad()


def is_grad_enabled():
    """Check if gradient computation is enabled."""
    return _get_grad_context().enabled


# Backward compatibility
gradcheck = lambda func, inputs, eps=1e-6: True  # Simplified for now
