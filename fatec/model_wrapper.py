"""
Model Wrapper for FATE-C Universal Training System
Provides a consistent interface for both build system and traditional models
"""

import numpy as np
from typing import Dict, List, Any, Optional
from .core.tensor import Tensor


class ModelWrapper:
    """
    Wrapper class that provides a consistent interface for trained models
    Compatible with both build system models and traditional Sequential models
    """
    
    def __init__(self, network, config=None):
        """
        Initialize model wrapper
        
        Args:
            network: The actual neural network (Sequential, etc.)
            config: Optional configuration dictionary
        """
        self.network = network
        self.config = config or {}
        self.training_history = {}
        self.is_trained = False
        self.optimizer = None
        self.loss_fn = None
    
    def compile(self, optimizer=None, loss=None, metrics=None):
        """
        Compile the model with optimizer and loss function
        
        Args:
            optimizer: Optimizer object or string
            loss: Loss function object or string
            metrics: List of metrics to track
        """
        if optimizer:
            self.optimizer = optimizer
        if loss:
            self.loss_fn = loss
        self.metrics = metrics or []
        
        print(f"✅ Model compiled with {type(self.network).__name__}")
        return self
    
    def predict(self, X):
        """
        Make predictions with the model
        
        Args:
            X: Input data (numpy array or Tensor)
        
        Returns:
            Predictions as numpy array
        """
        if not isinstance(X, Tensor):
            X = Tensor(X)
        
        predictions = self.network(X)
        return predictions.data
    
    def evaluate(self, X, y, metrics=None):
        """
        Evaluate the model on test data
        
        Args:
            X: Test features
            y: Test labels
            metrics: Metrics to calculate
        
        Returns:
            Dictionary of evaluation results
        """
        from .train import evaluate
        return evaluate(self, X, y, metrics or ["accuracy", "loss"])
    
    def fit(self, X, y, epochs=10, batch_size=32, **kwargs):
        """
        Train the model (convenience method)
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of epochs
            batch_size: Batch size
            **kwargs: Additional training arguments
        
        Returns:
            Training history
        """
        from .train import train
        trained_model = train(
            self, X, y, 
            epochs=epochs, 
            batch_size=batch_size,
            **kwargs
        )
        return trained_model.training_history
    
    def save(self, filepath):
        """Save model to file (placeholder)"""
        print(f"💾 Model saved to {filepath} (placeholder)")
    
    def load(self, filepath):
        """Load model from file (placeholder)"""
        print(f"📂 Model loaded from {filepath} (placeholder)")
    
    def summary(self):
        """Print model summary"""
        print(f"📋 Model Summary: {type(self.network).__name__}")
        if hasattr(self.network, 'layers'):
            print(f"   Layers: {len(self.network.layers)}")
            for i, layer in enumerate(self.network.layers):
                print(f"     {i+1}. {type(layer).__name__}")
        
        if self.training_history:
            print(f"   Training History: {list(self.training_history.keys())}")
        
        print(f"   Trained: {'✅' if self.is_trained else '❌'}")
    
    def visualize(self, mode="summary"):
        """Visualize model or training progress"""
        if mode == "summary":
            self.summary()
        elif mode == "history" and self.training_history:
            print("📊 Training History:")
            for metric, values in self.training_history.items():
                print(f"   {metric}: {values[-3:]}...")  # Show last 3 values
        else:
            print(f"Visualization mode '{mode}' not implemented")
    
    def to(self, backend):
        """Export to different backend (placeholder)"""
        print(f"🔄 Exporting to {backend} (placeholder)")
        return self
    
    def __call__(self, X):
        """Make the wrapper callable"""
        return self.predict(X)
    
    def __repr__(self):
        status = "trained" if self.is_trained else "untrained"
        return f"ModelWrapper({type(self.network).__name__}, {status})"
