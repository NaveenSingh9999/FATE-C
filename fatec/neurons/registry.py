"""Neuron registry system."""

_NEURON_REGISTRY = {}


def neuron(func=None, *, params=None):
    """Decorator to register a neuron function."""
    def decorator(f):
        name = f.__name__
        _NEURON_REGISTRY[name] = {
            'func': f,
            'params': params or {},
        }
        return f
    
    if func is None:
        return decorator
    else:
        return decorator(func)


def get_neuron(name):
    """Get neuron information by name."""
    return _NEURON_REGISTRY.get(name)
