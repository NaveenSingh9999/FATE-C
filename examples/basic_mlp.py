"""
Basic MLP Example with FATE-C Build System
Demonstrates the new universal build() API with a complete training example.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import fatec as fc

def generate_sample_data():
    """Generate sample data for demonstration."""
    # Generate synthetic MNIST-like data
    np.random.seed(42)
    num_samples = 1000
    input_dim = 784  # 28x28 flattened
    num_classes = 10
    
    # Generate random input features
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    
    # Generate random labels (one-hot encoded)
    y_labels = np.random.randint(0, num_classes, num_samples)
    y = np.eye(num_classes)[y_labels].astype(np.float32)
    
    # Split into train/test
    split_idx = int(0.8 * num_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Generated dataset:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Testing: {X_test.shape[0]} samples") 
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {num_classes}")
    
    return (X_train, y_train), (X_test, y_test)

def demo_task_builder():
    """Demonstrate task template builder."""
    print("\n🎯 TASK TEMPLATE DEMO")
    print("=" * 50)
    
    # Single line classifier using task template
    model = fc.build("task", 
                     name="classifier", 
                     input_dim=784, 
                     num_classes=10,
                     hidden_layers=[256, 128],
                     activation="relu",
                     dropout_rate=0.2)
    
    print("Task Template Model:")
    model.visualize("summary")
    return model

def demo_custom_builder():
    """Demonstrate custom network builder."""
    print("\n🛠️ CUSTOM NETWORK DEMO")
    print("=" * 50)
    
    # Custom network with intelligent neurons
    model = fc.seq([
        fc.Dense(128, activation="smart_relu"),    # Intelligent ReLU
        fc.Dropout(0.2),
        fc.Dense(64, activation="adaptive_sigmoid"), # Adaptive Sigmoid
        fc.Dropout(0.1),
        fc.Dense(10, activation="softmax")
    ])
    
    print("Custom Network with Intelligent Neurons:")
    model.visualize("summary")
    return model

def demo_training_workflow(model, train_data, test_data):
    """Demonstrate complete training workflow."""
    print("\n🚀 TRAINING DEMO")
    print("=" * 50)
    
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    # Convert to FATE-C tensors
    X_train_tensor = fc.Tensor(X_train)
    y_train_tensor = fc.Tensor(y_train)
    X_test_tensor = fc.Tensor(X_test)
    
    print(f"Model before compilation: {model}")
    
    # Compile model
    model.compile(
        optimizer="adam",
        loss="cross_entropy",
        learning_rate=0.001,
        metrics=["accuracy"]
    )
    
    print(f"Model after compilation: {model}")
    
    # Test prediction before training
    print("\nTesting model prediction...")
    try:
        sample_input = X_train_tensor[:5]  # First 5 samples
        prediction = model.predict(sample_input)
        print(f"Input shape: {sample_input.shape}")
        print(f"Prediction shape: {prediction.shape}")
        print("✅ Model prediction working!")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
    
    print("\n📊 Model ready for training!")
    print("   Training data shape:", X_train.shape)
    print("   Training labels shape:", y_train.shape)
    print("   Model architecture: Complete")
    print("   Optimizer: Adam")
    print("   Loss: Cross Entropy")
    print("   Ready for: model.fit(train_data, epochs=10)")

def main():
    """Main demonstration function."""
    print("🌟 BASIC MLP WITH FATE-C BUILD SYSTEM")
    print("=" * 60)
    
    # Generate sample data
    train_data, test_data = generate_sample_data()
    
    # Demo 1: Task template
    task_model = demo_task_builder()
    
    # Demo 2: Custom network
    custom_model = demo_custom_builder()
    
    # Demo 3: Training workflow
    demo_training_workflow(custom_model, train_data, test_data)
    
    print("\n🎉 BASIC MLP DEMO COMPLETE!")
    print("=" * 60)
    print("✨ Key Features Demonstrated:")
    print("   🎯 Task template: Quick classifier creation")
    print("   🛠️ Custom network: Intelligent neuron integration")
    print("   🚀 Training workflow: Complete model preparation")
    print("   📊 Data handling: Synthetic dataset generation")
    print("=" * 60)

if __name__ == "__main__":
    main()