"""
FATE-C Universal Training Demo
Demonstrates both fatec.train() and traditional training methods
"""

import numpy as np
import fatec as fc

def generate_sample_data():
    """Generate sample classification data"""
    np.random.seed(42)
    n_samples, n_features, n_classes = 100, 20, 3
    
    # Generate random data
    X = np.random.randn(n_samples, n_features)
    
    # Create separable classes
    y_raw = np.random.randint(0, n_classes, n_samples)
    y = np.eye(n_classes)[y_raw]  # One-hot encoding
    
    return X, y, y_raw

def demo_universal_training():
    """Demonstrate the new fatec.train() universal training function"""
    print("🌟" + "="*60)
    print("🚀 UNIVERSAL TRAINING WITH fatec.train()")
    print("="*60)
    
    # Generate data
    X, y, y_raw = generate_sample_data()
    print(f"📊 Data shape: {X.shape}, Labels shape: {y.shape}")
    
    # Method 1: Train a build system model
    print("\n🔧 Method 1: Build System + Universal Training")
    print("-" * 40)
    
    # Build model with build system
    model1 = fc.build("network", layers=[
        {"type": "Dense", "units": 64, "activation": "relu"},
        {"type": "Dense", "units": 32, "activation": "relu"},
        {"type": "Dense", "units": 3, "activation": "softmax"}
    ])
    
    print(f"📦 Built model: {type(model1).__name__}")
    
    # Train with universal function
    trained_model1 = fc.train(
        model1,
        data=X,
        labels=y,
        epochs=5,
        optimizer="adam",
        loss="cross_entropy",
        batch_size=16,
        metrics=["loss", "accuracy"],
        learning_rate=0.01
    )
    
    print(f"✅ Training history: {list(trained_model1.training_history.keys())}")
    
    # Test the model
    print("\n🧪 Testing Universal Training")
    print("-" * 40)
    
    # Generate test data
    X_test, y_test, _ = generate_sample_data()
    
    # Evaluate model
    results = fc.evaluate(trained_model1, X_test, y_test, metrics=["accuracy", "loss"])
    print(f"📈 Test Results - Accuracy: {results['accuracy']:.4f}, Loss: {results['loss']:.4f}")
    
    return trained_model1

if __name__ == "__main__":
    print("🌟 FATE-C Universal Training System Demo")
    print("=" * 70)
    
    # Demo new universal training
    trained_model = demo_universal_training()
    
    print("\n" + "="*70)
    print("🎉 Demo completed! fatec.train() is working! 🚀")
