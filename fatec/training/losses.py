"""Loss functions."""

import numpy as np


def mse_loss(y_pred, y_true):
    """Mean squared error loss."""
    return np.mean((y_pred - y_true) ** 2)


def auto_loss(y_pred, y_true):
    """Auto select loss function."""
    return mse_loss(y_pred, y_true)
