# FATE-C Learning Hub 🎓

Welcome to the FATE-C learning center! Master both the **Universal Build System** and **Traditional Approaches**.

## 🚀 Quick Start Guide

### Option 1: Universal Build System (Recommended for beginners)
```python
import fatec as fc

# One-liner classifier
model = fc.build("task", name="classifier", input_dim=20, num_classes=5)
model.compile()
# model.fit(train_data, epochs=10)
```

### Option 2: Traditional Approach (Full control)
```python
import fatec as fc

# Manual construction
model = fc.Sequential([fc.Dense(64, activation='relu'), fc.Dense(5)])
trainer = fc.Trainer(model, fc.Adam(), fc.get_loss("cross_entropy"))
# trainer.fit(X, y, epochs=10)
```

## 📚 Learning Path

### 🌟 Foundation Level
- **[01. Foundation Basics](01_foundation.md)** - **START HERE!** Both Build System and Traditional
- [02. Neuron Design](02_neurons.md) - Custom activation functions and intelligent neurons  
- [03. Layer Architecture](03_layers.md) - Building layers both ways

### 🔧 Intermediate Level  
- [04. Model Building](04_models.md) - Sequential vs custom models
- [05. Training & Optimization](05_training.md) - Both training approaches
- [06. Graph Models](06_graphs.md) - DAG models and complex architectures

### ⚡ Advanced Level
- [07. Multi-Model Systems](07_multi_models.md) - GANs and orchestration
- [08. Export & Interop](08_export.md) - PyTorch, TensorFlow export
- [09. Performance & Debugging](09_performance.md) - Optimization and profiling

## 🎯 Learning Approaches

### For Beginners: Build System First
```python
# Start here - one line models
model = fc.build("task", name="classifier", input_dim=784, num_classes=10)

# Progress to custom configs  
model = fc.build("network", layers=[
    {"type": "Dense", "units": 128, "activation": "smart_relu"},
    {"type": "Dense", "units": 10, "activation": "softmax"}
])

# Master the shorthand
model = fc.seq([
    fc.Dense(128, activation="relu"),
    fc.Dense(10, activation="softmax")
])
```

### For Experts: Traditional Methods
```python
# Manual layer construction
class CustomModel(fc.BaseModel):
    def __init__(self):
        super().__init__()
        self.layer1 = fc.Dense(128)
        self.activation = fc.SmartReLU()
        self.output = fc.Dense(10)
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        return self.output(x)

# Custom training loops
for epoch in range(epochs):
    for batch in dataloader:
        loss = custom_loss(model(batch.x), batch.y)
        loss.backward()
        optimizer.step()
```

## 🧪 Hands-On Examples

### Quick Test - Build System
```python
import fatec as fc
import numpy as np

# Generate data
X = np.random.randn(100, 20)
y = np.random.randint(0, 3, (100, 3))

# Build and train
model = fc.build("task", name="quick_test", input_dim=20, num_classes=3)
model.compile()
print("Build system ready! ✅")
```

### Quick Test - Traditional  
```python
import fatec as fc
from fatec.training.losses import get_loss

# Manual setup
model = fc.Sequential([fc.Dense(32, activation='relu'), fc.Dense(3)])
trainer = fc.Trainer(model, fc.Adam(), get_loss("cross_entropy"))
print("Traditional setup ready! ✅")
```

## 🎭 Hybrid Approach (Best of Both)
```python
# Start with build system
base = fc.build("task", name="base", input_dim=20, num_classes=5)

# Customize with traditional methods
network = base.network
custom_layer = fc.Dense(32, activation="smart_relu")
# Insert custom layer (conceptual)

# Train with build system
base.compile()
# base.fit(data, epochs=10)
```

## 📖 Documentation Structure

| File | Focus | Build System | Traditional | Level |
|------|-------|--------------|-------------|-------|
| [01_foundation.md](01_foundation.md) | **Core concepts** | ✅ Complete | ✅ Complete | Beginner |
| [02_neurons.md](02_neurons.md) | Activation functions | ✅ Formula API | ✅ Custom classes | Beginner |
| [03_layers.md](03_layers.md) | Layer construction | ✅ Config-driven | ✅ Manual building | Intermediate |
| [04_models.md](04_models.md) | Model architecture | ✅ Templates | ✅ Custom models | Intermediate |
| [05_training.md](05_training.md) | Training workflows | ✅ Auto-compile | ✅ Manual loops | Intermediate |

## 🏃‍♂️ Quick Examples

### 5-Minute Classifier
```python
import fatec as fc
model = fc.build("task", name="classifier", input_dim=784, num_classes=10)
model.compile()
model.visualize("summary")
```

### 5-Minute Custom Network
```python
import fatec as fc
model = fc.seq([
    fc.Dense(128, activation="smart_relu"),
    fc.Dropout(0.2),
    fc.Dense(10, activation="softmax")
])
model.compile()
```

## 🎯 Recommendations

- **New to ML?** → Start with Build System (01_foundation.md)
- **PyTorch/TF user?** → Check Traditional approach examples  
- **Research focus?** → Learn both, use Traditional for novel architectures
- **Production deployment?** → Master Build System for consistent results
- **Teaching ML?** → Use Build System for concepts, Traditional for internals

## 🚀 Next Steps

1. **Start:** [01. Foundation Basics](01_foundation.md) - Learn both approaches
2. **Practice:** Run `examples/training_demo.py` to see both in action
3. **Experiment:** Try `examples/build_demo.py` for build system features
4. **Build:** Create your own models using your preferred approach

**Remember:** FATE-C gives you the best of both worlds - simple when you need it, powerful when you want it! 🌟
