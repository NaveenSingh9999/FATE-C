"""FATE-C Universal Build System.

The build() function is the main entry point for creating neurons, layers, networks, and tasks.
This provides a unified, config-driven API that hides complexity while maintaining flexibility.
"""

import numpy as np
import warnings
from typing import Dict, List, Any, Union, Optional, Callable


class BuildError(Exception):
    """Exception raised when build() encounters errors."""
    pass


class Neuron:
    """Wrapper for individual neuron functions."""
    
    def __init__(self, func, name=None, formula=None):
        self.func = func
        self.name = name or func.__name__ if hasattr(func, '__name__') else 'neuron'
        self.formula = formula
        
    def __call__(self, x):
        return self.func(x)
    
    def __repr__(self):
        if self.formula:
            return f"Neuron(formula='{self.formula}')"
        return f"Neuron(name='{self.name}')"


class Model:
    """High-level model wrapper with training capabilities."""
    
    def __init__(self, network, name=None):
        self.network = network
        self.name = name or 'Model'
        self.optimizer = None
        self.loss_fn = None
        self.metrics = []
        self.trainer = None
        self._compiled = False
        
    def compile(self, optimizer='adam', loss='mse', metrics=None, learning_rate=0.001):
        """Compile the model with optimizer, loss, and metrics."""
        from .optim.adam import Adam
        from .optim.sgd import SGD
        from .training.losses import get_loss
        from .training.trainer import Trainer
        
        # Set up optimizer
        if isinstance(optimizer, str):
            if optimizer.lower() == 'adam':
                self.optimizer = Adam(learning_rate=learning_rate)
            elif optimizer.lower() == 'sgd':
                self.optimizer = SGD(learning_rate=learning_rate)
            else:
                raise BuildError(f"Unknown optimizer: {optimizer}")
        else:
            self.optimizer = optimizer
            
        # Set up loss function
        if isinstance(loss, str):
            self.loss_fn = get_loss(loss)
        else:
            self.loss_fn = loss
            
        # Set up metrics
        if metrics is None:
            metrics = []
        elif isinstance(metrics, str):
            metrics = [metrics]
        self.metrics = metrics
        
        # Create trainer
        self.trainer = Trainer(
            model=self.network,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn
        )
        
        self._compiled = True
        
    def fit(self, train_data, val_data=None, epochs=10, batch_size=32, verbose=True):
        """Train the model."""
        if not self._compiled:
            self.compile()  # Auto-compile with defaults
            
        return self.trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
    
    def predict(self, x):
        """Make predictions."""
        return self.network(x)
    
    def evaluate(self, test_data):
        """Evaluate the model."""
        if not self._compiled:
            raise BuildError("Model must be compiled before evaluation")
            
        return self.trainer.evaluate(test_data)
    
    def to(self, backend):
        """Export model to different backend."""
        print(f"Exporting to {backend} (not yet implemented)")
        return self
    
    def visualize(self, format='summary'):
        """Visualize the model."""
        if format == 'summary':
            return self._visualize_summary()
        else:
            return self._visualize_graph()
    
    def _visualize_graph(self):
        """Generate graph visualization."""
        print(f"Model: {self.name}")
        print("=" * 50)
        if hasattr(self.network, 'layers'):
            for i, layer in enumerate(self.network.layers):
                print(f"Layer {i+1}: {layer}")
        else:
            print(f"Network: {self.network}")
            
    def _visualize_summary(self):
        """Generate model summary."""
        total_params = 0
        print(f"Model: {self.name}")
        print("=" * 80)
        print(f"{'Layer':<20} {'Output Shape':<20} {'Params':<15} {'Type':<20}")
        print("-" * 80)
        
        if hasattr(self.network, 'layers'):
            for i, layer in enumerate(self.network.layers):
                params = layer.count_params() if hasattr(layer, 'count_params') else 0
                total_params += params
                output_shape = getattr(layer, '_output_shape', 'Unknown')
                layer_type = layer.__class__.__name__
                print(f"{layer.name:<20} {str(output_shape):<20} {params:<15,} {layer_type:<20}")
        
        print("-" * 80)
        print(f"Total params: {total_params:,}")
        print("=" * 80)
        
        return total_params
    
    def save(self, filepath):
        """Save model to file."""
        print(f"Saving model to {filepath} (not yet implemented)")
    
    def __repr__(self):
        compiled_str = "✓" if self._compiled else "✗"
        return f"Model(name='{self.name}', compiled={compiled_str})"


def build(mode: str, **kwargs) -> Union['Neuron', Any]:
    """
    Universal build function for FATE-C.
    
    Args:
        mode: Build mode - 'neuron', 'layer', 'network', or 'task'
        **kwargs: Configuration parameters specific to the mode
        
    Returns:
        Neuron, Layer, or Model object depending on mode
    """
    mode = mode.lower()
    
    if mode == "neuron":
        return _build_neuron(**kwargs)
    elif mode == "layer":
        return _build_layer(**kwargs)
    elif mode == "network":
        return _build_network(**kwargs)
    elif mode == "task":
        return _build_task(**kwargs)
    else:
        raise BuildError(f"Unknown build mode: {mode}. Use 'neuron', 'layer', 'network', or 'task'.")


def _build_neuron(formula: str = None, func: Callable = None, name: str = None) -> Neuron:
    """Build a neuron from formula or function."""
    if formula and func:
        raise BuildError("Specify either 'formula' or 'func', not both")
    
    if formula:
        # Parse and compile formula
        compiled_func = _compile_formula(formula)
        return Neuron(compiled_func, name=name, formula=formula)
    elif func:
        return Neuron(func, name=name)
    else:
        raise BuildError("Must specify either 'formula' or 'func' for neuron")


def _compile_formula(formula: str) -> Callable:
    """Compile a formula string into a callable function."""
    from .neurons.activations import get_activation
    
    # Simple formula compiler - can be extended
    formula = formula.strip()
    
    # Replace common math functions
    replacements = {
        'relu': '_relu',
        'tanh': '_tanh', 
        'sigmoid': '_sigmoid',
        'softmax': '_softmax'
    }
    
    compiled_formula = formula
    for old, new in replacements.items():
        compiled_formula = compiled_formula.replace(old, new)
    
    # Define helper functions
    def _relu(x):
        return get_activation('relu')(x)
    
    def _tanh(x):
        return get_activation('tanh')(x)
    
    def _sigmoid(x):
        return get_activation('sigmoid')(x)
    
    def _softmax(x):
        return get_activation('softmax')(x)
    
    # Create namespace for evaluation
    namespace = {
        '_relu': _relu,
        '_tanh': _tanh,
        '_sigmoid': _sigmoid,
        '_softmax': _softmax,
        'np': np,
        'exp': np.exp,
        'log': np.log,
        'sin': np.sin,
        'cos': np.cos,
        'abs': np.abs,
        'sqrt': np.sqrt,
        'max': np.maximum,
        'min': np.minimum
    }
    
    try:
        # Create function
        func_code = f"lambda x: {compiled_formula}"
        compiled_func = eval(func_code, namespace)
        return compiled_func
    except Exception as e:
        raise BuildError(f"Failed to compile formula '{formula}': {e}")


def _build_layer(type: str, **kwargs):
    """Build a layer of specified type."""
    from .layers.dense import Dense
    from .layers.dropout import Dropout
    
    layer_type = type.lower()
    
    if layer_type == "dense":
        return Dense(**kwargs)
    elif layer_type == "dropout":
        return Dropout(**kwargs)
    else:
        raise BuildError(f"Unknown layer type: {type}")


def _build_network(layers: List[Dict[str, Any]], name: str = None):
    """Build a network from layer specifications."""
    from .models.sequential import Sequential
    
    if not layers:
        raise BuildError("Network must have at least one layer")
    
    # Build layers
    built_layers = []
    for i, layer_config in enumerate(layers):
        if not isinstance(layer_config, dict):
            raise BuildError(f"Layer {i} config must be a dictionary")
        
        if 'type' not in layer_config:
            raise BuildError(f"Layer {i} must specify 'type'")
        
        layer_type = layer_config.pop('type')
        layer = _build_layer(layer_type, **layer_config)
        built_layers.append(layer)
    
    # Create sequential model
    network = Sequential(built_layers, name=name)
    return Model(network, name=name)


def _build_task(name: str, **kwargs):
    """Build a model from task template."""
    task_name = name.lower()
    
    if task_name == "classifier":
        return _build_classifier_task(**kwargs)
    elif task_name == "regressor":
        return _build_regressor_task(**kwargs)
    else:
        raise BuildError(f"Unknown task template: {name}")


def _build_classifier_task(input_dim: int, num_classes: int, 
                          hidden_layers: List[int] = None,
                          activation: str = 'relu',
                          dropout_rate: float = 0.0,
                          name: str = None):
    """Build a classification model template."""
    if hidden_layers is None:
        hidden_layers = [128, 64]
    
    layers = []
    
    # Hidden layers
    for units in hidden_layers:
        layers.append({
            "type": "Dense",
            "units": units,
            "activation": activation
        })
        if dropout_rate > 0:
            layers.append({
                "type": "Dropout",
                "rate": dropout_rate
            })
    
    # Output layer
    layers.append({
        "type": "Dense",
        "units": num_classes,
        "activation": "softmax" if num_classes > 2 else "sigmoid"
    })
    
    return _build_network(layers, name=name or "Classifier")


def _build_regressor_task(input_dim: int, output_dim: int = 1,
                         hidden_layers: List[int] = None,
                         activation: str = 'relu',
                         dropout_rate: float = 0.0,
                         name: str = None):
    """Build a regression model template."""
    if hidden_layers is None:
        hidden_layers = [128, 64]
    
    layers = []
    
    # Hidden layers
    for units in hidden_layers:
        layers.append({
            "type": "Dense", 
            "units": units,
            "activation": activation
        })
        if dropout_rate > 0:
            layers.append({
                "type": "Dropout",
                "rate": dropout_rate
            })
    
    # Output layer (no activation for regression)
    layers.append({
        "type": "Dense",
        "units": output_dim
    })
    
    return _build_network(layers, name=name or "Regressor")


# Convenience functions for common patterns
def seq(layers: List, name: str = None):
    """Create a sequential model from layers (shorthand)."""
    from .models.sequential import Sequential
    
    network = Sequential(layers, name=name)
    return Model(network, name=name or "SequentialModel")


def Dense(units: int, activation: str = None, **kwargs):
    """Shorthand for creating Dense layer."""
    from .layers.dense import Dense as DenseLayer
    return DenseLayer(units=units, activation=activation, **kwargs)


def Dropout(rate: float, **kwargs):
    """Shorthand for creating Dropout layer.""" 
    from .layers.dropout import Dropout as DropoutLayer
    return DropoutLayer(rate=rate, **kwargs)
