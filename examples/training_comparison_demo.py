"""
FATE-C Training Methods Comparison
Shows fatec.train() vs Traditional Trainer side by side
"""

import numpy as np
import fatec as fc

def generate_data():
    """Generate sample data for comparison"""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, (100, 3))  # One-hot encoded
    return X, y

def demo_universal_train():
    """Demonstrate fatec.train() universal training"""
    print("🚀 UNIVERSAL TRAINING: fatec.train()")
    print("-" * 50)
    
    X, y = generate_data()
    
    # Build model
    model = fc.build("network", layers=[
        {"type": "Dense", "units": 32, "activation": "relu"},
        {"type": "Dense", "units": 3, "activation": "softmax"}
    ])
    
    # Universal training - one function call!
    trained_model = fc.train(
        model, 
        data=X, 
        labels=y,
        epochs=3,
        optimizer="adam",
        loss="cross_entropy",
        batch_size=16,
        metrics=["loss", "accuracy"],
        learning_rate=0.01
    )
    
    print(f"✅ Done! History: {list(trained_model.training_history.keys())}")
    return trained_model

def demo_traditional_train():
    """Demonstrate traditional training approach"""
    print("\n🏗️ TRADITIONAL TRAINING: Manual Setup")
    print("-" * 50)
    
    X, y = generate_data()
    
    # Build model
    model = fc.Sequential([
        fc.Dense(32, activation='relu'),
        fc.Dense(3, activation='softmax')
    ])
    
    # Manual setup
    optimizer = fc.Adam(learning_rate=0.01)
    loss_fn = fc.training.losses.get_loss("cross_entropy")
    trainer = fc.Trainer(model, optimizer, loss_fn)
    
    print(f"🔧 Setup: {type(trainer).__name__} + {type(optimizer).__name__}")
    
    # Manual training
    history = trainer.fit(X, y, epochs=3, batch_size=16)
    
    print(f"✅ Done! History: {list(history.keys())}")
    return model, history

def compare_approaches():
    """Compare both training approaches"""
    print("\n" + "="*70)
    print("🔍 TRAINING APPROACHES COMPARISON")
    print("="*70)
    
    # Universal training
    universal_model = demo_universal_train()
    
    # Traditional training
    traditional_model, traditional_history = demo_traditional_train()
    
    # Comparison summary
    print("\n📋 COMPARISON SUMMARY")
    print("-" * 30)
    print("🚀 Universal Training (fatec.train):")
    print("   ✅ Simple API - just one function call")
    print("   ✅ Automatic metrics tracking")
    print("   ✅ Built-in progress display")
    print("   ✅ Works with both build system and traditional models")
    print("   ✅ Consistent interface")
    
    print("\n🏗️ Traditional Training:")
    print("   ✅ Full control over training loop")
    print("   ✅ Custom gradient computations")
    print("   ✅ Fine-grained debugging")
    print("   ✅ Research-friendly")
    print("   ✅ Extensible")
    
    print("\n💡 Recommendations:")
    print("   • Quick prototyping? → Use fatec.train()")
    print("   • Research/custom logic? → Use Traditional")
    print("   • Both approaches can coexist! 🤝")

if __name__ == "__main__":
    print("🌟 FATE-C: Universal vs Traditional Training")
    compare_approaches()
    print("\n🎉 Both methods demonstrated successfully! 🚀")
