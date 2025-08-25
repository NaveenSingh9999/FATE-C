"""Custom neuron example."""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import fatec as fc

# Define custom neuron
@fc.neuron
def swish(x):
    return x / (1 + np.exp(-x))

# Generate data
X = np.random.randn(50, 3)
y = np.random.randn(50, 1)

# Create model with custom neuron
model = fc.seq([
    fc.Dense(16, activation='relu'),
    fc.Dense(1)
])

model.compile()
model.fit(X, y, epochs=2)

print("Custom neuron example completed!")
