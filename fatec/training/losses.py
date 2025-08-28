"""Loss functions."""

import numpy as np


def mse_loss(y_pred, y_true):
    """Mean squared error loss."""
    return np.mean((y_pred - y_true) ** 2)


def cross_entropy_loss(y_pred, y_true):
    """Cross entropy loss."""
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred))


def auto_loss(y_pred, y_true):
    """Auto select loss function."""
    return mse_loss(y_pred, y_true)


def get_loss(loss_name):
    """Get loss function by name."""
    loss_name = loss_name.lower()
    
    if loss_name in ['mse', 'mean_squared_error']:
        return mse_loss
    elif loss_name in ['cross_entropy', 'crossentropy', 'categorical_crossentropy']:
        return cross_entropy_loss
    elif loss_name == 'auto':
        return auto_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
