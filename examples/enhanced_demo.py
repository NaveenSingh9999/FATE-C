"""
Enhanced FATE-C Demo: CSV/TXT Training with 97%+ Intelligence

Demonstrates FATE-C's enhanced capabilities with diverse data formats.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import fatec as fc
import numpy as np

print("Enhanced FATE-C Demo: Intelligent Training")
print("=" * 50)

# Create sample CSV data
def create_sample_csv():
    print("Creating sample CSV...")
    data = ["age,income,score,target"]
    
    for i in range(100):
        age = np.random.randint(20, 70)
        income = np.random.normal(50000, 10000)
        score = np.random.uniform(0, 100)
        target = 1.0 if (age * 0.01 + score * 0.005) > 1.5 else 0.0
        
        data.append(f"{age},{income:.0f},{score:.1f},{target}")
    
    with open("demo.csv", 'w') as f:
        f.write('\n'.join(data))
    
    print("Created demo.csv with 100 samples")
    return "demo.csv"

# Create sample TXT data
def create_sample_txt():
    print("Creating sample TXT...")
    texts = [
        "This is amazing and wonderful!",
        "Terrible and awful experience.",
        "Good quality and nice design.",
        "Bad service and poor results.",
        "Excellent work and great job!",
        "Disappointing and frustrating.",
        "Perfect solution for my needs.",
        "Horrible quality and waste of money."
    ]
    
    with open("demo.txt", 'w') as f:
        f.write('\n'.join(texts))
    
    print("Created demo.txt with 8 samples")
    return "demo.txt"

# Demo CSV training
def demo_csv():
    print("\n" + "=" * 30)
    print("CSV Training Demo")
    print("=" * 30)
    
    csv_file = create_sample_csv()
    
    # Create model with intelligent neurons
    model = fc.Sequential([
        fc.Dense(8, activation='smart_relu'),
        fc.Dense(4, activation='adaptive_sigmoid'),
        fc.Dense(1, activation='sigmoid')
    ])
    
    # Enhanced trainer targeting 97%+
    trainer = fc.EnhancedTrainer(model, target_accuracy=0.97)
    
    # Train from CSV
    history = trainer.train_from_csv(csv_file, epochs=30)
    
    print(f"Results:")
    print(f"Accuracy: {history['accuracy'][-1]:.3f}")
    print(f"Intelligence: {history['intelligence_score'][-1]:.1f}%")
    
    summary = trainer.get_performance_summary()
    print(f"Target Achieved: {'✓' if summary['target_achieved'] else '✗'}")

# Demo TXT training
def demo_txt():
    print("\n" + "=" * 30)
    print("TXT/NLP Training Demo")
    print("=" * 30)
    
    txt_file = create_sample_txt()
    
    # NLP model
    model = fc.Sequential([
        fc.Dense(10, activation='smart_relu'),
        fc.Dense(5, activation='adaptive_sigmoid'),
        fc.Dense(1, activation='sigmoid')
    ])
    
    trainer = fc.EnhancedTrainer(model, target_accuracy=0.97)
    
    # Train from TXT
    history = trainer.train_from_txt(txt_file, epochs=25)
    
    print(f"NLP Results:")
    print(f"Accuracy: {history['accuracy'][-1]:.3f}")
    print(f"Intelligence: {history['intelligence_score'][-1]:.1f}%")
    
    summary = trainer.get_performance_summary()
    print(f"Target Achieved: {'✓' if summary['target_achieved'] else '✗'}")

# Demo intelligent neurons
def demo_neurons():
    print("\n" + "=" * 30)
    print("Intelligent Neurons Demo")
    print("=" * 30)
    
    # Test intelligent neurons
    smart_relu = fc.SmartReLU()
    adaptive_sigmoid = fc.AdaptiveSigmoid()
    
    test_data = fc.Tensor(np.random.randn(50, 5))
    
    relu_out = smart_relu(test_data)
    sigmoid_out = adaptive_sigmoid(test_data)
    
    print(f"Smart ReLU Intelligence: {smart_relu.get_intelligence_score():.1f}%")
    print(f"Adaptive Sigmoid Intelligence: {adaptive_sigmoid.get_intelligence_score():.1f}%")

# Main demo
if __name__ == "__main__":
    demo_neurons()
    demo_csv()
    demo_txt()
    
    print("\n" + "=" * 50)
    print("Enhanced FATE-C Features:")
    print("✓ CSV/TXT data loading")
    print("✓ Intelligent neurons (97%+ target)")
    print("✓ Enhanced training with quality monitoring")
    print("✓ Performance scoring and optimization")
    print("=" * 50)
    
    # Cleanup
    try:
        os.remove("demo.csv")
        os.remove("demo.txt")
    except:
        pass
