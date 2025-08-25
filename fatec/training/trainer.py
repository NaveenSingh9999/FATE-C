"""Training engine."""

import numpy as np
from ..core.tensor import Tensor


class Trainer:
    """Training engine for FATE-C models."""
    
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    def fit(self, x, y, epochs=1, batch_size=32, val=None):
        """Train the model."""
        x = np.array(x)
        y = np.array(y)
        
        n_samples = len(x)
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch_x = x[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                # Forward pass
                predictions = self.model(batch_x)
                
                # Compute loss
                if hasattr(predictions, 'data'):
                    pred_data = predictions.data
                else:
                    pred_data = predictions
                
                loss = self.loss_fn(pred_data, batch_y)
                total_loss += loss
                n_batches += 1
                
                # Simple gradient update (placeholder)
                parameters = self.model.get_parameters()
                gradients = {}
                
                for name, param in parameters.items():
                    gradients[name] = np.random.randn(*param.data.shape) * 0.01
                
                self.optimizer.step(parameters, gradients)
            
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return {"loss": avg_loss}
