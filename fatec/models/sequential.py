"""Sequential model."""

from .base import BaseModel


class Sequential(BaseModel):
    """Sequential model - layers applied in order."""
    
    def __init__(self, layers=None, name=None):
        super().__init__(name)
        self.layers = layers or []
    
    def add(self, layer):
        """Add a layer to the model."""
        self.layers.append(layer)
    
    def forward(self, x):
        """Forward pass through all layers."""
        output = x
        for layer in self.layers:
            output = layer(output)
        return output
    
    def get_parameters(self):
        """Get all trainable parameters from all layers."""
        params = {}
        for i, layer in enumerate(self.layers):
            layer_params = layer.get_parameters()
            for name, param in layer_params.items():
                params[f"layer_{i}_{name}"] = param
        return params
