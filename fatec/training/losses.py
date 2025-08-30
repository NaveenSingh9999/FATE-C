"""Loss functions."""

import numpy as np
from ..core.tensor import Tensor


def mse_loss(y_pred, y_true):
    """Mean squared error loss."""
    return np.mean((y_pred - y_true) ** 2)


def cross_entropy_loss(y_pred, y_true):
    """Cross entropy loss."""
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred))


def tensor_mse_loss(y_pred, y_true):
    """Mean squared error loss for tensors."""
    if hasattr(y_pred, 'data'):
        y_pred_data = y_pred.data
    else:
        y_pred_data = y_pred
    
    if hasattr(y_true, 'data'):
        y_true_data = y_true.data
    else:
        y_true_data = y_true
    
    diff = y_pred_data - y_true_data
    loss_val = np.mean(diff ** 2)
    return Tensor(loss_val)


def tensor_cross_entropy_loss(y_pred, y_true):
    """Cross entropy loss for tensors."""
    if hasattr(y_pred, 'data'):
        y_pred_data = y_pred.data
    else:
        y_pred_data = y_pred
    
    if hasattr(y_true, 'data'):
        y_true_data = y_true.data
    else:
        y_true_data = y_true
    
    # Clip predictions to prevent log(0)
    y_pred_clipped = np.clip(y_pred_data, 1e-15, 1 - 1e-15)
    loss_val = -np.mean(y_true_data * np.log(y_pred_clipped))
    return Tensor(loss_val)


def auto_loss(y_pred, y_true):
    """Auto select loss function."""
    return mse_loss(y_pred, y_true)


def get_loss(loss_name):
    """Get loss function by name."""
    loss_name = loss_name.lower()
    
    if loss_name in ['mse', 'mean_squared_error']:
        return tensor_mse_loss
    elif loss_name in ['cross_entropy', 'crossentropy', 'categorical_crossentropy']:
        return tensor_cross_entropy_loss
    elif loss_name == 'auto':
        return auto_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# Legacy functions for backward compatibility
def get_numpy_loss(loss_name):
    """Get numpy-based loss function by name."""
    loss_name = loss_name.lower()
    
    if loss_name in ['mse', 'mean_squared_error']:
        return mse_loss
    elif loss_name in ['cross_entropy', 'crossentropy', 'categorical_crossentropy']:
        return cross_entropy_loss
    elif loss_name == 'auto':
        return auto_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
