"""Test tutorial examples."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import fatec as fc
import numpy as np

print("Testing Foundation Tutorial...")

# Test tensor operations
t1 = fc.Tensor([1, 2, 3])
t2 = fc.Tensor([4, 5, 6])
result = t1 + t2
print(f"Tensor add: {result.data}")

# Test model
model = fc.seq([fc.Dense(8, activation='relu'), fc.Dense(1)])
X = np.random.randn(20, 5)
y = np.random.randn(20, 1)

model.compile()
history = model.fit(X, y, epochs=2)
print(f"Training loss: {history['loss']:.4f}")

print("Tutorial examples work! ✅")
