# 01. Foundation Basics - Your First Steps with FATE-C

Welcome to FATE-C! This tutorial teaches you everything about the foundation we've built, including the revolutionary **Universal Build System**.

## What is FATE-C?

FATE-C is a **neuron-first** neural network framework that lets you:
- Design networks in **1 line** with the Universal Build System
- Create **custom neurons** with simple decorators  
- Build with **traditional layers** or **intelligent neurons**
- **Export to major frameworks** (PyTorch, TensorFlow, JAX, ONNX)
- Progress from **simple to complex** architectures seamlessly

## Core Philosophy

```
One API to rule them all: fatec.build()
```

- **Universal Build**: `fc.build()` creates neurons, layers, networks, and complete tasks
- **Dual Approach**: Traditional layer-by-layer OR config-driven templates
- **Auto-inference**: Shapes, optimizers, loss functions figured out automatically
- **Progressive**: Start with templates, customize with traditional methods
- **Portable**: Design once, run anywhere

---

## Part 1: The Universal Build System 🚀

### Quick Start - Single Line Models

```python
import fatec as fc

# 🎯 TASK TEMPLATES - One line complete models
classifier = fc.build("task", 
                     name="classifier",
                     input_dim=784, 
                     num_classes=10)

regressor = fc.build("task",
                    name="regressor", 
                    input_dim=20,
                    output_dim=1)

# ✅ Ready to train!
classifier.compile()
# classifier.fit(train_data, epochs=10)
```

### Build System Modes

The `fc.build()` function has 4 modes:

#### 1. **Neuron Mode** - Custom Activation Functions
```python
# Create neurons from mathematical formulas
relu_neuron = fc.build("neuron", formula="relu(x)")
custom_neuron = fc.build("neuron", formula="tanh(x) + 0.1*x")  # Leaky tanh

# Test them
x = fc.Tensor([[1.0, -1.0, 0.5]])
print(relu_neuron(x))     # [[1.0, 0.0, 0.5]]
print(custom_neuron(x))   # Custom activation output
```

#### 2. **Layer Mode** - Individual Layer Creation
```python
# Build layers with configuration
dense = fc.build("layer", type="Dense", units=64, activation="relu")
dropout = fc.build("layer", type="Dropout", rate=0.3)

# Use in networks
x = fc.Tensor([[1, 2, 3, 4]])
output = dense(x)  # Shape: (1, 64)
```

#### 3. **Network Mode** - Complete Neural Networks  
```python
# Build networks from layer specifications
model = fc.build("network", layers=[
    {"type": "Dense", "units": 128, "activation": "relu"},
    {"type": "Dropout", "rate": 0.2},
    {"type": "Dense", "units": 64, "activation": "relu"}, 
    {"type": "Dense", "units": 10, "activation": "softmax"}
], name="MyClassifier")

# Visualize architecture
model.visualize("summary")
```

#### 4. **Task Mode** - Pre-configured Templates
```python
# Quick classifier template
mnist_model = fc.build("task",
                      name="classifier",
                      input_dim=784,
                      num_classes=10,
                      hidden_layers=[256, 128],
                      activation="relu", 
                      dropout_rate=0.2)

# Quick regressor template
boston_model = fc.build("task",
                       name="regressor",
                       input_dim=13,
                       output_dim=1,
                       hidden_layers=[64, 32])
```

### Shorthand API ⚡

For even faster prototyping:

```python
# Sequential models with shorthand
quick_model = fc.seq([
    fc.Dense(128, activation="relu"),
    fc.Dropout(0.2),
    fc.Dense(10, activation="softmax")
])

# Compile and train
quick_model.compile(optimizer="adam", loss="cross_entropy")
# quick_model.fit(data, epochs=5)
```

---

## Part 2: Traditional Layer-by-Layer Approach 🏗️

For fine-grained control, use the traditional approach:

### Understanding Tensors

Tensors are the basic data containers in FATE-C - smart NumPy arrays with gradient tracking.

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

## Part 3: Traditional Layers - Building Blocks 🧱

### Dense (Fully Connected) Layers

The most fundamental layer type:

```python
# Traditional approach - Manual layer creation
layer = fc.Dense(units=64, activation='relu', use_bias=True)

# Apply to data
x = fc.Tensor(np.random.randn(32, 10))  # 32 samples, 10 features
output = layer(x)

print(f"Input shape: {x.shape}")     # (32, 10)
print(f"Output shape: {output.shape}")  # (32, 64)

# Access parameters
params = layer.get_parameters()
print(f"Parameters: {list(params.keys())}")  # ['weight', 'bias']
```

### Sequential Models - Traditional Way

```python
# Build step by step
model = fc.Sequential([
    fc.Dense(128, activation='relu'),
    fc.Dropout(0.2),
    fc.Dense(64, activation='relu'),
    fc.Dense(10, activation='softmax')
])

# Manual training setup
optimizer = fc.Adam(learning_rate=0.001)
trainer = fc.Trainer(model=model, optimizer=optimizer, loss_fn=fc.mse_loss)

# Train manually
# trainer.fit(train_x, train_y, epochs=10)
```

### Intelligent Neurons - Advanced Traditional

```python
# Create intelligent neurons manually
smart_relu = fc.SmartReLU(adaptation_rate=0.01)
adaptive_sigmoid = fc.AdaptiveSigmoid(initial_scale=1.0)

# Use in manual layer construction
class CustomLayer(fc.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.smart_activation = smart_relu
    
    def forward(self, x):
        linear_output = x @ self.weights + self.bias
        return self.smart_activation(linear_output)
```

---

## Part 4: Training - Both Approaches 🏋️‍♂️

### 🚀 Build System Training (Recommended)

**Easy Mode - Automatic Everything:**
```python
# Generate sample data
X = np.random.randn(1000, 20)
y = np.random.randint(0, 3, (1000, 3))  # One-hot encoded

# One-liner model creation
model = fc.build("task", name="classifier", 
                 input_dim=20, num_classes=3)

# Auto-compile with sensible defaults
model.compile()  # Uses Adam optimizer, cross_entropy loss

# Auto-fit with progress tracking
history = model.fit(
    train_data=(X[:800], y[:800]),
    val_data=(X[800:], y[800:]),
    epochs=20,
    batch_size=32
)

# Evaluate and export
accuracy = model.evaluate(test_data)
model.to("pytorch")  # Export to PyTorch
```

**Custom Mode - Full Control:**
```python
# Custom configuration
model = fc.build("network", layers=[
    {"type": "Dense", "units": 128, "activation": "smart_relu"},
    {"type": "Dropout", "rate": 0.3},
    {"type": "Dense", "units": 64, "activation": "adaptive_sigmoid"},
    {"type": "Dense", "units": 3, "activation": "softmax"}
])

# Custom compilation
model.compile(
    optimizer="adam", 
    loss="cross_entropy",
    learning_rate=0.001,
    metrics=["accuracy"]
)

# Advanced training with callbacks
history = model.fit(
    train_data=train_data,
    val_data=val_data,
    epochs=50,
    batch_size=64,
    verbose=True
)
```

### 🏗️ Traditional Training (Expert Mode)

**Manual Setup:**
```python
# Manual model construction
model = fc.Sequential([
    fc.Dense(128, activation='relu'),
    fc.Dropout(0.2), 
    fc.Dense(64, activation='relu'),
    fc.Dense(10, activation='softmax')
])

# Manual optimizer and loss
optimizer = fc.Adam(learning_rate=0.001)
loss_fn = fc.cross_entropy_loss

# Manual trainer creation
trainer = fc.Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn
)

# Manual training loop
for epoch in range(20):
    # Custom batch processing
    for batch_x, batch_y in data_loader:
        # Forward pass
        predictions = model(batch_x)
        
        # Compute loss
        loss = loss_fn(predictions, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"Epoch {epoch+1} completed")
```

**Enhanced Training with Intelligence:**
```python
# Use enhanced trainer for intelligent features
enhanced_trainer = fc.EnhancedTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    intelligence_threshold=0.95,
    target_accuracy=0.97
)

# Train with intelligence monitoring
results = enhanced_trainer.train(
    train_data=train_data,
    val_data=val_data,
    epochs=100,
    batch_size=32,
    intelligent_stopping=True
)

print(f"Intelligence Score: {results['intelligence_score']}")
print(f"Target Achieved: {results['target_achieved']}")
```

---

## Part 5: Comparison - When to Use What? 🤔

### Use Build System When:
✅ **Quick Prototyping** - Need models fast  
✅ **Standard Architectures** - Common patterns (classifier, regressor)  
✅ **Beginner Friendly** - Learning neural networks  
✅ **Production Deployment** - Need consistent, validated architectures  
✅ **Template-based Work** - Similar models with different parameters  

### Use Traditional When:
✅ **Research & Experimentation** - Novel architectures  
✅ **Fine-grained Control** - Custom layer implementations  
✅ **Educational Purposes** - Understanding internals  
✅ **Complex Workflows** - Multi-model systems  
✅ **Custom Training Logic** - Non-standard training procedures  

### Hybrid Approach (Best of Both):
```python
# Start with build system for base architecture
base_model = fc.build("task", name="classifier", 
                     input_dim=784, num_classes=10)

# Extract the network for customization
network = base_model.network

# Add custom layers traditionally
custom_layer = fc.Dense(32, activation="smart_relu")
network.add(custom_layer)

# Use build system for training
base_model.compile()
base_model.fit(train_data, epochs=10)
```

---

## Part 6: Complete Examples 📚

### Example 1: Image Classification (Build System)
```python
import fatec as fc
import numpy as np

# Simulate MNIST data
X_train = np.random.randn(1000, 784)
y_train = np.random.randint(0, 10, (1000, 10))  # One-hot

# One-liner classifier
model = fc.build("task", name="mnist_classifier",
                 input_dim=784, num_classes=10,
                 hidden_layers=[512, 256, 128],
                 activation="relu", dropout_rate=0.2)

# Train
model.compile(optimizer="adam", learning_rate=0.001)
history = model.fit((X_train, y_train), epochs=20)

# Visualize
model.visualize("summary")
```

### Example 2: Custom Architecture (Traditional)
```python
import fatec as fc

# Custom intelligent network
class IntelligentMLP(fc.BaseModel):
    def __init__(self):
        super().__init__()
        self.layers = [
            fc.Dense(128, activation=None),  # No activation yet
            fc.SmartReLU(adaptation_rate=0.02),  # Intelligent activation
            fc.Dropout(0.3),
            fc.Dense(64, activation=None),
            fc.AdaptiveSigmoid(initial_scale=2.0),  # Another intelligent neuron
            fc.Dense(10, activation='softmax')
        ]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Create and train
model = IntelligentMLP()
trainer = fc.EnhancedTrainer(model, fc.Adam(), fc.cross_entropy_loss)
trainer.train(train_data, epochs=50)
```

### Example 3: Hybrid Approach
```python
# Start with template, customize with traditional methods
model = fc.build("task", name="custom_classifier",
                 input_dim=100, num_classes=5)

# Access internal network for modifications
network = model.network

# Insert custom intelligent layer
smart_layer = fc.Dense(32, activation="smart_relu")
network.layers.insert(-1, smart_layer)  # Before final layer

# Train with build system
model.compile()
model.fit(train_data, epochs=30)
```

---

## Summary: The Power of Dual Approaches 🎯

FATE-C gives you **the best of both worlds**:

### 🚀 Build System Benefits:
- **1-line model creation** with `fc.build()`
- **Auto-configuration** of optimizers and losses
- **Template-based** rapid prototyping
- **Production-ready** architectures
- **Beginner-friendly** API

### 🏗️ Traditional Benefits:
- **Fine-grained control** over every component
- **Research flexibility** for novel architectures  
- **Educational transparency** for learning
- **Custom implementations** of layers and training
- **Expert-level** optimization

### 🎭 Intelligent Features (Both):
- **Smart neurons** that adapt during training
- **Enhanced training** with intelligence monitoring
- **Target-driven** learning (97%+ accuracy goals)
- **Automatic optimization** based on performance

**Next Steps:**
1. Try the build system for quick wins
2. Explore traditional methods for deeper understanding
3. Combine both approaches for maximum flexibility
4. Experiment with intelligent neurons for cutting-edge performance

**Remember:** Simple tasks deserve simple solutions, complex problems need powerful tools. FATE-C provides both! 🌟
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
