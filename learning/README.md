# FATE-C Learning Hub

Welcome to the FATE-C learning center!

## Table of Contents

### Foundation Level
- [01. Foundation Basics](01_foundation.md) - Core concepts and your first neural network
- [02. Neuron Design](02_neurons.md) - Custom activation functions
- [03. Layer Architecture](03_layers.md) - Building layers

### Intermediate Level  
- [04. Model Building](04_models.md) - Sequential models
- [05. Training & Optimization](05_training.md) - Training loops and optimizers
- [06. Graph Models](06_graphs.md) - DAG models

### Advanced Level
- [07. Multi-Model Systems](07_multi_models.md) - GANs and orchestration
- [08. Export & Interop](08_export.md) - PyTorch, TensorFlow export
- [09. Performance & Debugging](09_performance.md) - Optimization

## Quick Start

```python
import fatec as fc
import numpy as np

X = np.random.randn(100, 4)
y = np.random.randn(100, 1)

model = fc.seq([
    fc.Dense(64, activation='relu'),
    fc.Dense(1)
])

model.compile()
model.fit(X, y, epochs=3)
```

Start with [Foundation Basics](01_foundation.md)!
