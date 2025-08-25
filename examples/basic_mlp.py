"""Basic MLP example."""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import fatec as fc

# Generate sample data
X = np.random.randn(100, 4)
y = np.random.randn(100, 1)

# Create model
model = fc.seq([
    fc.Dense(64, activation='relu'),
    fc.Dense(32, activation='relu'),
    fc.Dense(1)
])

# Compile and train
model.compile()
history = model.fit(X, y, epochs=3)

print("Training completed!")
print(f"Final loss: {history['loss']:.4f}")
