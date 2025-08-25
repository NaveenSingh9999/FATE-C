# 01. Foundation Basics - Your First Steps with FATE-C

Welcome to FATE-C! This tutorial teaches you everything about the foundation we've built.

## What is FATE-C?

FATE-C is a **neuron-first** neural network framework that lets you:
- Design networks in **10-20 lines** of pure Python
- Create **custom neurons** with simple decorators
- **Export to major frameworks** (PyTorch, TensorFlow, JAX, ONNX)
- Build **simple to complex** architectures progressively

## Core Philosophy

```
Simple stays simple, complexity is opt-in
```

- **Flat API**: Everything accessible via `import fatec as fc`
- **Auto-inference**: Shapes, optimizers, loss functions figured out automatically
- **Progressive**: Start with Sequential, grow to Graphs and Multi-models
- **Portable**: Design once, run anywhere

---

## Part 1: Understanding Tensors

### What are Tensors?

Tensors are the basic data containers in FATE-C - think of them as smart NumPy arrays with gradient tracking.

```python
import fatec as fc
import numpy as np

# Create tensors
t1 = fc.Tensor([1, 2, 3])
t2 = fc.Tensor([[1, 2], [3, 4]])

print(f"Shape: {t1.shape}")  # Shape: (3,)
print(f"Data: {t1.data}")    # Underlying NumPy array
```

### Tensor Operations

```python
# Basic operations
a = fc.Tensor([1, 2, 3])
b = fc.Tensor([4, 5, 6])

result = a + b  # Addition
print(result.data)  # [5, 7, 9]

# Matrix operations
x = fc.Tensor([[1, 2], [3, 4]])
y = fc.Tensor([[5, 6], [7, 8]])

z = x @ y  # Matrix multiplication
print(z.shape)  # (2, 2)
```

### Gradient Tracking

```python
# Enable gradients for training
weights = fc.Tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
inputs = fc.Tensor([[1, 2]])

output = inputs @ weights
print(f"Requires grad: {output.requires_grad}")  # True
```

**Key Concepts:**
- `requires_grad=True` enables gradient computation
- Operations preserve gradient requirements
- Gradients flow backward through computations

---

## Part 2: Layers - Building Blocks

### Dense (Fully Connected) Layers

The most fundamental layer type:

```python
# Create a Dense layer
layer = fc.Dense(units=64, activation='relu', use_bias=True)

# Apply to data
x = fc.Tensor(np.random.randn(32, 10))  # 32 samples, 10 features
output = layer(x)

print(f"Input shape: {x.shape}")     # (32, 10)
print(f"Output shape: {output.shape}") # (32, 64)
```

**How it works:**
1. **Lazy Initialization**: Weights created on first call based on input shape
2. **Xavier Init**: Smart weight initialization for stable training
3. **Linear Transform**: `output = input @ weights + bias`
4. **Activation**: Optional function applied element-wise

### Layer Parameters

```python
# After first call, layer has parameters
print("Layer parameters:")
for name, param in layer.get_parameters().items():
    print(f"  {name}: {param.shape}")

# Outputs:
#   weight: (10, 64)
#   bias: (64,)

### Dropout for Regularization

```python
dropout = fc.Dropout(rate=0.5)

# During training
dropout.training = True
train_output = dropout(x)  # Randomly zeros 50% of inputs

# During evaluation  
dropout.training = False
eval_output = dropout(x)   # No dropout applied
```

### Available Activations

```python
# Built-in activations
fc.Dense(32, activation='relu')    # ReLU: max(0, x)
fc.Dense(32, activation='sigmoid') # Sigmoid: 1/(1+exp(-x))
fc.Dense(32, activation='tanh')    # Tanh: tanh(x)
fc.Dense(32, activation='gelu')    # GELU: Gaussian Error Linear Unit

# No activation (linear)
fc.Dense(32, activation=None)
```

---

## Part 3: Your First Neural Network

### The Sequential Model

The simplest way to build networks:

```python
# Method 1: Using fc.seq()
model = fc.seq([
    fc.Dense(128, activation='relu'),
    fc.Dense(64, activation='relu'), 
    fc.Dense(10, activation='relu'),
    fc.Dense(1)  # Output layer
])

# Method 2: Using Sequential class directly
model = fc.Sequential([
    fc.Dense(128, activation='relu'),
    fc.Dense(64, activation='relu'),
    fc.Dense(1)
])
```

### Understanding the Architecture

```python
# Let's trace what happens:
x = np.random.randn(100, 20)  # 100 samples, 20 features

print("Data flow through network:")
print(f"Input: {x.shape}")           # (100, 20)

# Layer 1: 20 -> 128
# Layer 2: 128 -> 64  
# Layer 3: 64 -> 1

output = model(x)
print(f"Output: {output.shape}")     # (100, 1)
```

---

## Part 4: Training Your Network

### Compilation - Setting Up Training

```python
# Compile with defaults (recommended for beginners)
model.compile()

# This automatically sets:
# - optimizer='adam' (adaptive learning rate)
# - loss='auto' (inferred from target data)
```

### The Training Process

```python
# Generate sample data
X_train = np.random.randn(1000, 20)  # Features
y_train = np.random.randn(1000, 1)   # Targets

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,        # Number of training passes
    batch_size=32     # Samples per update
)

print(f"Final loss: {history['loss']:.4f}")
```

**What happens during training:**
1. **Forward Pass**: Data flows through layers
2. **Loss Computation**: Compare predictions to targets
3. **Backward Pass**: Compute gradients
4. **Parameter Update**: Adjust weights using optimizer
5. **Repeat**: For all batches and epochs

### Making Predictions

```python
# Make predictions on new data
X_test = np.random.randn(50, 20)
predictions = model.predict(X_test)

print(f"Predictions shape: {predictions.shape}")  # (50, 1)

---

## Part 5: Complete Examples

### Example 1: Regression (Predicting Continuous Values)

```python
import fatec as fc
import numpy as np

# Generate regression data
np.random.seed(42)
X = np.random.randn(500, 5)
y = X[:, 0] + 2*X[:, 1] - X[:, 2] + 0.1*np.random.randn(500)
y = y.reshape(-1, 1)

# Build model
model = fc.seq([
    fc.Dense(64, activation='relu'),
    fc.Dense(32, activation='relu'),
    fc.Dense(1)  # No activation for regression
])

# Train
model.compile()
model.fit(X, y, epochs=20, batch_size=32)

# Evaluate
test_X = np.random.randn(10, 5)
predictions = model.predict(test_X)
print(f"Test predictions: {predictions.flatten()}")
```

### Example 2: Custom Neurons

```python
import fatec as fc
import numpy as np

# Define custom neuron
@fc.neuron
def swish(x):
    """Swish activation: x * sigmoid(x)"""
    return x / (1 + np.exp(-x))

# Use in model
model = fc.seq([
    fc.Dense(64, activation='relu'),
    fc.Dense(32, activation=swish),  # Custom neuron
    fc.Dense(1)
])

# Generate data and train
X = np.random.randn(200, 8)
y = np.random.randn(200, 1)

model.compile()
model.fit(X, y, epochs=10)
print("Custom neuron model trained successfully!")
```

### Example 3: Multi-layer with Dropout

```python
import fatec as fc
import numpy as np

# Create dataset
np.random.seed(123)
X = np.random.randn(800, 10)
y = np.sum(X[:, :3], axis=1, keepdims=True) + 0.1*np.random.randn(800, 1)

# Build model with regularization
model = fc.seq([
    fc.Dense(128, activation='relu'),
    fc.Dropout(0.3),
    fc.Dense(64, activation='relu'),
    fc.Dropout(0.2),
    fc.Dense(32, activation='relu'),
    fc.Dense(1)
])

# Train
model.compile()
history = model.fit(X, y, epochs=15, batch_size=32)

print(f"Final loss: {history['loss']:.4f}")
```

---

## Part 6: Best Practices

### 1. Data Preparation

```python
# Normalize your features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std
```

### 2. Architecture Design

```python
# Good starting architecture:
model = fc.seq([
    fc.Dense(input_dim * 2, activation='relu'),  # First layer: 2x input size
    fc.Dropout(0.2),                             # Light regularization
    fc.Dense(input_dim, activation='relu'),      # Second layer: 1x input size
    fc.Dense(output_dim)                         # Output layer
])
```

### 3. Training Tips

```python
# Start with defaults
model.compile()  # Use Adam optimizer and auto loss

# If overfitting (loss not decreasing):
# - Add more dropout
# - Reduce model size
# - Get more data

# If underfitting (loss stays high):
# - Increase model size
# - Train longer
# - Reduce dropout
```

---

## Part 7: Debugging Common Issues

### Issue 1: Shape Mismatch Errors

```python
# Error: Matrix multiplication shape mismatch
# Solution: Check your data shapes

print(f"Input shape: {X.shape}")
print(f"Expected: (batch_size, n_features)")

# Make sure data is 2D
if X.ndim == 1:
    X = X.reshape(-1, 1)
```

### Issue 2: Import Errors

```python
# If you get "No module named 'fatec'"
import sys
import os
sys.path.append('/path/to/FATE-C')
import fatec as fc
```

### Debugging Tools

```python
# Check model parameters
params = model.get_parameters()
for name, param in params.items():
    print(f"{name}: {param.shape}")
```

---

## Summary

FATE-C makes neural network creation **simple and progressive**:

```python
# From this simple start...
model = fc.seq([fc.Dense(64, activation='relu'), fc.Dense(1)])
model.compile()
model.fit(X, y, epochs=10)

# ...to unlimited complexity as you grow!
```

### What You've Learned

✅ **Tensors**: Basic data containers and operations  
✅ **Layers**: Dense layers, activations, dropout  
✅ **Models**: Sequential architecture building  
✅ **Training**: Compilation, fitting, and prediction  
✅ **Custom Neurons**: Creating new activation functions  
✅ **Best Practices**: Data prep, architecture, debugging  

### Next Steps

1. Try the examples in `/examples/` directory
2. Experiment with your own datasets
3. Explore advanced tutorials (coming soon)

**Ready for more?** Check out the other tutorials in the learning directory!

---

*Questions or issues? Examine the working examples in `/examples/basic_mlp.py` and `/examples/custom_neuron.py`.*
```
```
