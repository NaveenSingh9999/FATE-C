"""Basic tests for FATE-C."""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import fatec as fc


def test_tensor_creation():
    """Test tensor creation."""
    t1 = fc.Tensor([1, 2, 3])
    assert t1.shape == (3,)
    print("✓ Tensor test passed")


def test_dense_layer():
    """Test Dense layer."""
    layer = fc.Dense(3, activation='relu')
    x = fc.Tensor(np.random.randn(2, 4))
    output = layer(x)
    assert output.shape == (2, 3)
    print("✓ Dense layer test passed")


if __name__ == "__main__":
    print("Running FATE-C tests...")
    test_tensor_creation()
    test_dense_layer()
    print("Tests completed! ✅")
