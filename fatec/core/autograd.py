"""Autograd engine for FATE-C."""

import numpy as np
from typing import List


class Variable:
    """Variable node in the computation graph."""
    
    def __init__(self, data, grad_fn=None, requires_grad=True):
        self.data = np.asarray(data)
        self.grad_fn = grad_fn
        self.requires_grad = requires_grad
        self.grad = None
    
    def backward(self, grad=None):
        """Compute gradients via backpropagation."""
        if not self.requires_grad:
            return
        
        if grad is None:
            grad = np.ones_like(self.data)
        
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad
        
        if self.grad_fn:
            input_grads = self.grad_fn.backward(grad)
            for inp, inp_grad in zip(self.grad_fn.inputs, input_grads):
                if inp.requires_grad:
                    inp.backward(inp_grad)


class Function:
    """Base class for differentiable functions."""
    
    def __init__(self):
        self.inputs = []
    
    def forward(self, *inputs):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError
    
    def __call__(self, *inputs):
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
        
        output_data = self.forward(*[v.data for v in var_inputs])
        
        if requires_grad:
            output = Variable(output_data, grad_fn=self, requires_grad=True)
        else:
            output = Variable(output_data, requires_grad=False)
        
        return output


class no_grad:
    """Context manager to disable gradient computation."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def grad(outputs, inputs):
    """Compute gradients of outputs with respect to inputs."""
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    
    for inp in inputs:
        inp.grad = None
    
    for output in outputs:
        output.backward()
    
    return [inp.grad for inp in inputs]
