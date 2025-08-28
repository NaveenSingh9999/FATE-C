"""
FATE-C Build System Demo - Complete Universal API
Demonstrates the power of fatec.build() for creating neurons, layers, networks, and tasks.
"""

import numpy as np
import fatec as fc

def demo_neuron_creation():
    """Demonstrate creating custom neurons with formulas."""
    print("🧠 NEURON CREATION")
    print("=" * 40)
    
    # Create neurons using formulas
    relu_neuron = fc.build("neuron", formula="relu(x)")
    custom_neuron = fc.build("neuron", formula="tanh(x) + 0.1*x")  # Leaky tanh
    
    # Test them
    x = fc.Tensor([[1.0, -1.0, 0.5, -0.5]])
    
    print(f"Input: {x.data}")
    print(f"ReLU: {relu_neuron(x).data}")
    print(f"Leaky Tanh: {custom_neuron(x).data}")
    print()


def demo_layer_building():
    """Demonstrate building individual layers."""
    print("🏗️ LAYER BUILDING")
    print("=" * 40)
    
    # Build layers with different configurations
    dense = fc.build("layer", type="Dense", units=64, activation="relu")
    dropout = fc.build("layer", type="Dropout", rate=0.3)
    
    print(f"Dense Layer: {dense}")
    print(f"Dropout Layer: {dropout}")
    print()


def demo_network_creation():
    """Demonstrate creating complete networks."""
    print("🌐 NETWORK CREATION")
    print("=" * 40)
    
    # Method 1: Using build() with layer specs
    classifier = fc.build("network", layers=[
        {"type": "Dense", "units": 128, "activation": "relu"},
        {"type": "Dropout", "rate": 0.2},
        {"type": "Dense", "units": 64, "activation": "relu"},
        {"type": "Dense", "units": 3, "activation": "softmax"}
    ], name="MyClassifier")
    
    print("Method 1 - Layer Specifications:")
    classifier.visualize("summary")
    print()
    
    # Method 2: Using shorthand seq()
    regressor = fc.seq([
        fc.Dense(100, activation="relu"),
        fc.Dropout(0.1),
        fc.Dense(50, activation="relu"),
        fc.Dense(1)  # No activation for regression
    ])  # Remove name parameter for now
    
    print("Method 2 - Shorthand seq():")
    print(f"Type of regressor: {type(regressor)}")
    print(f"Regressor: {regressor}")
    
    if hasattr(regressor, 'visualize'):
        regressor.visualize("summary")
    else:
        print("No visualize method found")
    print()


def demo_task_templates():
    """Demonstrate using task templates."""
    print("📋 TASK TEMPLATES")
    print("=" * 40)
    
    # Quick classifier from template
    mnist_classifier = fc.build("task",
                                name="classifier",
                                input_dim=784,
                                num_classes=10,
                                hidden_layers=[256, 128],
                                activation="relu",
                                dropout_rate=0.2)
    
    print("MNIST Classifier Template:")
    mnist_classifier.visualize("summary")
    print()
    
    # Quick regressor from template  
    house_price_model = fc.build("task",
                                 name="regressor",
                                 input_dim=13,
                                 output_dim=1,
                                 hidden_layers=[64, 32])
    
    print("House Price Regressor Template:")
    house_price_model.visualize("summary")
    print()


def demo_training_workflow():
    """Demonstrate complete training workflow."""
    print("🚀 TRAINING WORKFLOW")
    print("=" * 40)
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(200, 4)  # 4 features
    # Create binary classification problem
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y_onehot = np.eye(2)[y]  # Convert to one-hot
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")
    
    # Build model using task template
    model = fc.build("task",
                     name="classifier", 
                     input_dim=4,
                     num_classes=2,
                     hidden_layers=[32, 16],
                     activation="relu",
                     dropout_rate=0.1)
    
    print(f"\nModel: {model}")
    
    # Compile with specific settings
    model.compile(
        optimizer="adam",
        loss="cross_entropy", 
        learning_rate=0.01,
        metrics=["accuracy"]
    )
    
    print(f"Compiled: {model}")
    print()
    
    # Show that model is ready for training
    print("✅ Model ready for training!")
    print("   - Use model.fit(train_data, epochs=10)")
    print("   - Use model.predict(new_data)")
    print("   - Use model.evaluate(test_data)")
    print("   - Use model.to('pytorch') for export")
    print()


def demo_intelligent_neurons():
    """Demonstrate integration with intelligent neurons."""
    print("🎯 INTELLIGENT NEURONS")
    print("=" * 40)
    
    # Create network with intelligent neurons
    smart_model = fc.seq([
        fc.Dense(64, activation="smart_relu"),     # Intelligent ReLU
        fc.Dense(32, activation="adaptive_sigmoid"), # Adaptive Sigmoid
        fc.Dense(10, activation="softmax")
    ], name="SmartModel")
    
    print("Smart Model with Intelligent Neurons:")
    smart_model.visualize("summary")
    print()
    
    # Test with some data
    test_input = fc.Tensor(np.random.randn(5, 10))
    print(f"Input shape: {test_input.shape}")
    
    try:
        output = smart_model.predict(test_input)
        print(f"Output shape: {output.shape}")
        print("✅ Intelligent neurons working!")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()


if __name__ == "__main__":
    print("🌟 FATE-C UNIVERSAL BUILD SYSTEM")
    print("=" * 60)
    print("The one API to rule them all: fatec.build()")
    print("=" * 60)
    print()
    
    try:
        demo_neuron_creation()
        demo_layer_building() 
        demo_network_creation()
        demo_task_templates()
        demo_training_workflow()
        demo_intelligent_neurons()
        
        print("🎉 ALL DEMOS COMPLETED!")
        print("=" * 60)
        print("✨ FATE-C Build System Features:")
        print("   🧠 Neuron creation from formulas")
        print("   🏗️ Layer building with configs") 
        print("   🌐 Network assembly from specs")
        print("   📋 Task templates for common problems")
        print("   ⚡ Shorthand API for quick prototyping")
        print("   🎯 Integration with intelligent neurons")
        print("   🚀 Production-ready training workflow")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()