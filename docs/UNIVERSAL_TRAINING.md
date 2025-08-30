# FATE-C Universal Training System

## 🚀 Overview

FATE-C now features **dual training approaches**:

1. **`fatec.train()`** - Universal training function (NEW!)
2. **Traditional Trainer** - Full control approach (existing)

Both methods coexist and serve different needs.

---

## �� Universal Training with `fatec.train()`

### Purpose
- Train models created with `fatec.build()` or `fatec.Sequential`
- Hide NumPy/backend loops from user
- Provide clean, universal training interface
- Support multiple backends and metrics

### Basic Usage

```python
import fatec as fc

# Build a model
model = fc.build("network", layers=[
    {"type": "Dense", "units": 128, "activation": "relu"},
    {"type": "Dense", "units": 10, "activation": "softmax"}
])

# Train it (one function call!)
trained_model = fc.train(
    model,
    data=train_X,
    labels=train_y,
    epochs=10,
    optimizer="adam",
    loss="cross_entropy",
    batch_size=32,
    metrics=["accuracy", "loss"]
)
```

## 🎯 When to Use Which?

### Use `fatec.train()` for:
- ✅ **Quick prototyping** - Get models running fast
- ✅ **Standard workflows** - Common training patterns
- ✅ **Consistent API** - Same interface for all models
- ✅ **Automatic metrics** - Built-in tracking and display
- ✅ **Beginners** - Simple, one-function approach

### Use Traditional Trainer for:
- ✅ **Research projects** - Custom training logic
- ✅ **Advanced optimization** - Custom gradient computation
- ✅ **Debugging** - Step-by-step control
- ✅ **Novel architectures** - Experimental setups
- ✅ **Performance tuning** - Fine-grained optimization

## 🎉 Summary

**FATE-C Universal Training System** provides:

✅ **`fatec.train()`** - Simple, universal training function
✅ **Traditional Trainer** - Full control approach  
✅ **Both methods coexist** - Choose what fits your needs
✅ **Seamless integration** - Works with entire FATE-C ecosystem
✅ **End-to-end pipeline** - From `fatec.build()` to `fatec.train()`

**First true end-to-end ML pipeline in FATE-C!** 🚀
