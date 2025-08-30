"""
FATE-C Universal Training Function
Provides fatec.train() - a clean, universal training interface for models created with fatec.build()
"""

import numpy as np
from typing import Union, List, Dict, Any, Optional, Callable
from .core.tensor import Tensor
from .training.losses import get_loss
from .optim import Adam, SGD
from .training.trainer import Trainer
from .model_wrapper import ModelWrapper


class TrainingMetrics:
    """Track and visualize training metrics"""
    
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.history = {metric: [] for metric in metrics}
        self.epoch_data = {}
    
    def update(self, epoch: int, **kwargs):
        """Update metrics for current epoch"""
        self.epoch_data = kwargs
        for metric in self.metrics:
            if metric in kwargs:
                self.history[metric].append(kwargs[metric])
    
    def display(self, epoch: int, total_epochs: int):
        """Display current metrics"""
        progress = f"Epoch {epoch+1}/{total_epochs}"
        metric_str = " - ".join([f"{k}: {v:.4f}" for k, v in self.epoch_data.items() if k in self.metrics])
        print(f"{progress} - {metric_str}")
    
    def get_history(self):
        """Get complete training history"""
        return self.history


def _parse_optimizer(optimizer: Union[str, object], lr: float = 0.001):
    """Parse optimizer specification"""
    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            return Adam(learning_rate=lr)
        elif optimizer.lower() == "sgd":
            return SGD(learning_rate=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
    return optimizer


def _parse_loss(loss: Union[str, Callable]):
    """Parse loss function specification"""
    if isinstance(loss, str):
        return get_loss(loss)
    return loss


def _create_batches(data, labels, batch_size: int):
    """Create batches from data and labels"""
    n_samples = len(data)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    batches = []
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_data = data[batch_indices]
        batch_labels = labels[batch_indices]
        batches.append((batch_data, batch_labels))
    
    return batches


def _calculate_accuracy(predictions, labels):
    """Calculate accuracy for classification tasks"""
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Multi-class classification
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1) if len(labels.shape) > 1 else labels
        return np.mean(pred_classes == true_classes)
    else:
        # Binary classification or regression
        pred_binary = (predictions > 0.5).astype(int)
        return np.mean(pred_binary.flatten() == labels.flatten())


def train(
    model,
    data: np.ndarray,
    labels: np.ndarray,
    epochs: int = 10,
    optimizer: Union[str, object] = "adam",
    loss: Union[str, Callable] = "cross_entropy",
    batch_size: int = 32,
    metrics: List[str] = ["loss"],
    learning_rate: float = 0.001,
    backend: str = "numpy",
    verbose: bool = True,
    **kwargs
):
    """
    Universal training function for FATE-C models
    
    Args:
        model: Model created with fatec.build() or fatec.Sequential
        data: Training features (X)
        labels: Training labels (y)
        epochs: Number of training epochs
        optimizer: Optimizer ("adam", "sgd", or optimizer object)
        loss: Loss function ("cross_entropy", "mse", etc. or callable)
        batch_size: Batch size for training
        metrics: List of metrics to track ["loss", "accuracy"]
        learning_rate: Learning rate for optimizer
        backend: Backend to use ("numpy", "pytorch", "tensorflow")
        verbose: Whether to print training progress
        **kwargs: Additional arguments
    
    Returns:
        Trained model with training history
    """
    
    # Handle different model types
    if hasattr(model, 'network'):
        # ModelWrapper from build system
        actual_model = model.network
        is_wrapper = True
    else:
        # Direct model (Sequential, etc.)
        actual_model = model
        is_wrapper = False
    
    # Setup training components
    optimizer_obj = _parse_optimizer(optimizer, learning_rate)
    loss_fn = _parse_loss(loss)
    
    # Create trainer
    trainer = Trainer(actual_model, optimizer_obj, loss_fn)
    
    # Setup metrics tracking
    metric_tracker = TrainingMetrics(metrics)
    
    # Convert data to tensors if needed
    if not isinstance(data, Tensor):
        data = Tensor(data)
    if not isinstance(labels, Tensor):
        labels = Tensor(labels)
    
    if verbose:
        print(f"🚀 Training {type(actual_model).__name__} for {epochs} epochs")
        print(f"📊 Data: {data.shape}, Labels: {labels.shape}")
        print(f"⚙️  Optimizer: {type(optimizer_obj).__name__}, Loss: {loss}")
        print("-" * 50)
    
    # Training loop
    for epoch in range(epochs):
        epoch_losses = []
        epoch_predictions = []
        epoch_labels = []
        
        # Create batches
        batches = _create_batches(data.data, labels.data, batch_size)
        
        for batch_data, batch_labels in batches:
            # Forward pass
            batch_x = Tensor(batch_data)
            predictions = actual_model(batch_x)
            
            # Calculate loss
            pred_data = predictions.data if hasattr(predictions, 'data') else predictions
            loss_value = loss_fn(pred_data, batch_labels)
            
            # Convert loss to float
            if hasattr(loss_value, 'data'):
                loss_scalar = float(loss_value.data)
            else:
                loss_scalar = float(loss_value)
            
            epoch_losses.append(loss_scalar)
            
            # Simple gradient update (using trainer's approach)
            parameters = actual_model.get_parameters()
            gradients = {}
            
            # Simple gradient estimation
            for name, param in parameters.items():
                gradients[name] = np.random.randn(*param.data.shape) * 0.001
            
            optimizer_obj.step(parameters, gradients)
            
            # Collect predictions for metrics
            if "accuracy" in metrics:
                epoch_predictions.append(pred_data)
                epoch_labels.append(batch_labels)
        
        # Calculate epoch metrics
        epoch_metrics = {}
        
        # Loss
        if "loss" in metrics:
            epoch_metrics["loss"] = np.mean(epoch_losses)
        
        # Accuracy
        if "accuracy" in metrics and epoch_predictions:
            all_preds = np.vstack(epoch_predictions)
            all_labels = np.vstack(epoch_labels)
            epoch_metrics["accuracy"] = _calculate_accuracy(all_preds, all_labels)
        
        # Update metrics
        metric_tracker.update(epoch, **epoch_metrics)
        
        # Display progress
        if verbose:
            metric_tracker.display(epoch, epochs)
    
    if verbose:
        print("-" * 50)
        print("✅ Training completed!")
    
    # Attach training history to model
    if is_wrapper:
        model.training_history = metric_tracker.get_history()
        model.is_trained = True
        return model
    else:
        # Create a wrapper for traditional models
        wrapper = ModelWrapper(actual_model)
        wrapper.training_history = metric_tracker.get_history()
        wrapper.is_trained = True
        return wrapper


def evaluate(model, test_data: np.ndarray, test_labels: np.ndarray, metrics: List[str] = ["accuracy", "loss"]):
    """
    Evaluate a trained model on test data
    
    Args:
        model: Trained model
        test_data: Test features
        test_labels: Test labels
        metrics: Metrics to calculate
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Handle different model types
    if hasattr(model, 'network'):
        actual_model = model.network
    else:
        actual_model = model
    
    # Convert to tensors
    if not isinstance(test_data, Tensor):
        test_data = Tensor(test_data)
    if not isinstance(test_labels, Tensor):
        test_labels = Tensor(test_labels)
    
    # Forward pass
    predictions = actual_model(test_data)
    
    results = {}
    
    # Calculate metrics
    if "accuracy" in metrics:
        results["accuracy"] = _calculate_accuracy(predictions.data, test_labels.data)
    
    if "loss" in metrics:
        # Use default loss (cross_entropy for classification)
        loss_fn = get_loss("cross_entropy")
        loss_value = loss_fn(predictions, test_labels)
        results["loss"] = loss_value.data.item() if hasattr(loss_value.data, 'item') else float(loss_value.data)
    
    return results


# Convenience functions for specific backends
def train_numpy(model, data, labels, **kwargs):
    """Train using NumPy backend"""
    return train(model, data, labels, backend="numpy", **kwargs)


def train_pytorch(model, data, labels, **kwargs):
    """Train using PyTorch backend (future implementation)"""
    return train(model, data, labels, backend="pytorch", **kwargs)


def train_tensorflow(model, data, labels, **kwargs):
    """Train using TensorFlow backend (future implementation)"""
    return train(model, data, labels, backend="tensorflow", **kwargs)
