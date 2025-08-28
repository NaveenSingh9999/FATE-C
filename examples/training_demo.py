"""
Advanced Training Demo - Build System vs Traditional
Demonstrates both approaches for neural network training in FATE-C.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import fatec as fc

def generate_data():
    """Generate sample data."""
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 5, (1000, 5))  # One-hot
    return X[:800], y[:800], X[800:], y[800:]

def demo_build_system():
    """Demonstrate Build System approach."""
    print("🚀 BUILD SYSTEM APPROACH")
    print("=" * 50)
    
    # Quick template
    model = fc.build("task", name="classifier",
                     input_dim=20, num_classes=5)
    print(f"Model: {model}")
    
    # Compile and test
    model.compile()
    print(f"Compiled: {model}")
    
    # Custom network
    custom = fc.seq([
        fc.Dense(64, activation="relu"),
        fc.Dense(5, activation="softmax")
    ])
    print(f"Custom: {custom}")
    
    return model

def demo_traditional():
    """Demonstrate Traditional approach."""
    print("\n🏗️ TRADITIONAL APPROACH")
    print("=" * 50)
    
    # Manual construction
    model = fc.Sequential([
        fc.Dense(64, activation='relu'),
        fc.Dense(5, activation='softmax')
    ])
    print(f"Manual model: {model}")
    
    # Manual trainer
    from fatec.training.losses import get_loss
    optimizer = fc.Adam(learning_rate=0.001)
    trainer = fc.Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=get_loss("cross_entropy")
    )
    print("Traditional trainer created")
    
    return model, trainer

def main():
    """Main demo."""
    print("🌟 TRAINING APPROACHES DEMO")
    print("=" * 60)
    
    # Generate data
    X_train, y_train, X_test, y_test = generate_data()
    print(f"Data: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    # Demo both approaches
    build_model = demo_build_system()
    traditional_model, trainer = demo_traditional()
    
    # Test predictions
    test_input = fc.Tensor(X_test[:3])
    
    print("\n🧪 Testing Predictions:")
    try:
        pred1 = build_model.predict(test_input)
        print(f"Build System: {pred1.shape} ✅")
    except Exception as e:
        print(f"Build System: {e}")
        
    try:
        pred2 = traditional_model(test_input)
        print(f"Traditional: {pred2.shape} ✅")
    except Exception as e:
        print(f"Traditional: {e}")
    
    print("\n✅ Demo complete!")
    print("Choose Build System for speed, Traditional for control!")

if __name__ == "__main__":
    main()