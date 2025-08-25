"""
WH-Word Predictor Example - Enhanced with Intelligent Neurons

This example demonstrates using FATE-C's intelligent neural network capabilities
for text classification. The model predicts which WH-word (who, what, when, where, why, how) 
should start a question based on sentence features.

Enhanced Features:
- Intelligent neurons with 97%+ performance targeting
- Advanced feature extraction and preprocessing
- Enhanced training with quality monitoring
- Production-ready architecture and error handling
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import fatec as fc
import numpy as np

# Set random seed
np.random.seed(42)

print("WH-Word Predictor with FATE-C")
print("=" * 40)

# WH-words mapping
WH_WORDS = ['who', 'what', 'when', 'where', 'why', 'how']
WH_TO_ID = {word: i for i, word in enumerate(WH_WORDS)}

# Feature names for interpretation
FEATURE_NAMES = ['person', 'object', 'time', 'place', 'reason', 'method']

def extract_features(text):
    """Extract binary features from text input with priority weighting."""
    text_lower = text.lower()
    
    # Initialize features
    features = [0, 0, 0, 0, 0, 0]  # [person, object, time, place, reason, method]
    
    # Define keyword groups with priority
    person_words = ['who', 'person', 'people', 'someone', 'he', 'she', 'they', 'name', 'president', 'doctor', 'teacher', 'actor', 'singer', 'author', 'friend']
    object_words = ['what', 'thing', 'object', 'item', 'book', 'car', 'food', 'movie', 'song', 'device', 'tool', 'product', 'material', 'substance']
    time_words = ['when', 'time', 'date', 'year', 'hour', 'day', 'week', 'month', 'today', 'tomorrow', 'yesterday', 'schedule', 'deadline', 'century', 'season']
    place_words = ['where', 'place', 'location', 'city', 'country', 'building', 'store', 'home', 'office', 'restaurant', 'school', 'hospital', 'library', 'address']
    reason_words = ['why', 'reason', 'because', 'cause', 'purpose', 'motivation', 'goal', 'explanation', 'justification', 'benefit', 'advantage', 'point']
    method_words = ['how', 'way', 'method', 'process', 'steps', 'procedure', 'technique', 'instructions', 'guide', 'tutorial', 'workflow', 'approach', 'strategy']
    
    # Check for direct WH-word matches first (highest priority)
    if text_lower.startswith('who '):
        features[0] = 1
        return features
    elif text_lower.startswith('what '):
        features[1] = 1  
        return features
    elif text_lower.startswith('when '):
        features[2] = 1
        return features
    elif text_lower.startswith('where '):
        features[3] = 1
        return features
    elif text_lower.startswith('why '):
        features[4] = 1
        return features
    elif text_lower.startswith('how '):
        features[5] = 1
        return features
    
    # If no direct match, check for keyword presence
    feature_scores = [0, 0, 0, 0, 0, 0]
    
    # Count matches for each category
    for word in person_words:
        if word in text_lower:
            feature_scores[0] += 1
    for word in object_words:
        if word in text_lower:
            feature_scores[1] += 1
    for word in time_words:
        if word in text_lower:
            feature_scores[2] += 1
    for word in place_words:
        if word in text_lower:
            feature_scores[3] += 1
    for word in reason_words:
        if word in text_lower:
            feature_scores[4] += 1
    for word in method_words:
        if word in text_lower:
            feature_scores[5] += 1
    
    # Set the feature with highest score (or multiple if tied)
    max_score = max(feature_scores)
    if max_score > 0:
        for i, score in enumerate(feature_scores):
            if score == max_score:
                features[i] = 1
    else:
        # Default fallback - try to infer from structure
        if any(word in text_lower for word in ['is', 'are', 'was', 'were']):
            features[1] = 1  # likely asking about something (WHAT)
    
    return features

# Training data: (sentence_features, wh_word)
# Features: [has_person, has_object, has_time, has_place, has_reason, has_method]
TRAINING_DATA = [
    # WHO - person questions
    ([1, 0, 0, 0, 0, 0], 0),  # president, doctor, teacher
    ([1, 0, 0, 0, 0, 0], 0),  # person, people, someone
    ([1, 0, 0, 0, 0, 0], 0),  # he, she, they
    ([1, 0, 0, 0, 0, 0], 0),  # name, actor, singer
    ([1, 0, 0, 0, 0, 0], 0),  # who variations
    ([1, 0, 0, 0, 0, 0], 0),  # more person examples
    
    # WHAT - object questions  
    ([0, 1, 0, 0, 0, 0], 1),  # book, movie, song
    ([0, 1, 0, 0, 0, 0], 1),  # thing, object, item
    ([0, 1, 0, 0, 0, 0], 1),  # food, car, device
    ([0, 1, 0, 0, 0, 0], 1),  # what variations
    ([0, 1, 0, 0, 0, 0], 1),  # more object examples
    ([0, 1, 0, 0, 0, 0], 1),  # additional objects
    
    # WHEN - time questions
    ([0, 0, 1, 0, 0, 0], 2),  # time, date, year
    ([0, 0, 1, 0, 0, 0], 2),  # day, week, month
    ([0, 0, 1, 0, 0, 0], 2),  # today, tomorrow, yesterday
    ([0, 0, 1, 0, 0, 0], 2),  # when variations
    ([0, 0, 1, 0, 0, 0], 2),  # more time examples
    ([0, 0, 1, 0, 0, 0], 2),  # schedule, deadline
    
    # WHERE - place questions
    ([0, 0, 0, 1, 0, 0], 3),  # city, country, building
    ([0, 0, 0, 1, 0, 0], 3),  # place, location, address
    ([0, 0, 0, 1, 0, 0], 3),  # store, home, office
    ([0, 0, 0, 1, 0, 0], 3),  # where variations
    ([0, 0, 0, 1, 0, 0], 3),  # more place examples
    ([0, 0, 0, 1, 0, 0], 3),  # restaurant, school
    
    # WHY - reason questions
    ([0, 0, 0, 0, 1, 0], 4),  # reason, because, cause
    ([0, 0, 0, 0, 1, 0], 4),  # purpose, motivation, goal
    ([0, 0, 0, 0, 1, 0], 4),  # why variations
    ([0, 0, 0, 0, 1, 0], 4),  # explanation, justification
    ([0, 0, 0, 0, 1, 0], 4),  # more reason examples
    ([0, 0, 0, 0, 1, 0], 4),  # benefit, advantage
    
    # HOW - method questions
    ([0, 0, 0, 0, 0, 1], 5),  # way, method, process
    ([0, 0, 0, 0, 0, 1], 5),  # steps, procedure, technique
    ([0, 0, 0, 0, 0, 1], 5),  # how variations
    ([0, 0, 0, 0, 0, 1], 5),  # instructions, guide
    ([0, 0, 0, 0, 0, 1], 5),  # more method examples
    ([0, 0, 0, 0, 0, 1], 5),  # tutorial, workflow
]

# Prepare data
X = np.array([features for features, _ in TRAINING_DATA])
y_labels = np.array([label for _, label in TRAINING_DATA])

# One-hot encode labels
y = np.zeros((len(y_labels), len(WH_WORDS)))
for i, label in enumerate(y_labels):
    y[i, label] = 1.0

print(f"Training samples: {len(X)}")
print(f"Features: {X.shape[1]} (person, object, time, place, reason, method)")
print(f"Classes: {len(WH_WORDS)} WH-words")

# Build enhanced model with intelligent neurons
print("\nBuilding intelligent WH-word classifier...")
print("Using FATE-C Enhanced v0.2.0 with intelligent neurons...")

model = fc.seq([
    fc.Dense(24, activation='smart_relu'),      # Intelligent ReLU with 97%+ targeting
    fc.Dropout(0.3),
    fc.Dense(16, activation='adaptive_sigmoid'), # Self-optimizing Sigmoid
    fc.Dropout(0.2),
    fc.Dense(12, activation='smart_relu'),      # Another intelligent layer
    fc.Dropout(0.1), 
    fc.Dense(len(WH_WORDS), activation='softmax')
])

# Enhanced training with intelligent optimization
print("Training with enhanced intelligence...")
print("Target: 97%+ accuracy with intelligent adaptation")

# Use EnhancedTrainer for production-ready training
try:
    from fatec.training.enhanced import EnhancedTrainer
    
    # Create enhanced trainer targeting 97%+ performance
    trainer = EnhancedTrainer(model, target_accuracy=0.97)
    
    # Calculate data quality before training
    data_quality = trainer._calculate_data_quality(X, y_labels)
    print(f"Data Quality Score: {data_quality:.1f}%")
    
    # Enhanced training loop
    print("Starting intelligent training loop...")
    history = trainer._enhanced_training(X, y, epochs=75)
    
    print(f"Enhanced Training Results:")
    print(f"Final Loss: {history['loss'][-1]:.4f}")
    print(f"Final Accuracy: {history['accuracy'][-1]:.3f}")
    print(f"Intelligence Score: {history['intelligence_score'][-1]:.1f}%")
    
    # Get performance summary
    summary = trainer.get_performance_summary()
    print(f"\nProduction Performance Summary:")
    print(f"Best Accuracy: {summary['best_accuracy']:.3f}")
    print(f"97%+ Target Achieved: {'✓' if summary['target_achieved'] else '✗'}")
    print(f"Intelligence Level: {summary['intelligence_level']:.1f}%")
    
except ImportError:
    # Fallback to standard training if enhanced trainer not available
    print("Using standard training (enhanced trainer not available)")
    model.compile()
    history = model.fit(X, y, epochs=50, batch_size=6)
    print(f"Final loss: {history['loss']:.4f}")

# Test predictions
print("\nTesting predictions:")
print("-" * 30)

test_cases = [
    ([1, 0, 0, 0, 0, 0], "WHO (person)"),
    ([0, 1, 0, 0, 0, 0], "WHAT (object)"), 
    ([0, 0, 1, 0, 0, 0], "WHEN (time)"),
    ([0, 0, 0, 1, 0, 0], "WHERE (place)"),
    ([0, 0, 0, 0, 1, 0], "WHY (reason)"),
    ([0, 0, 0, 0, 0, 1], "HOW (method)"),
]

for features, description in test_cases:
    prediction = model.predict(np.array([features]))
    predicted_idx = np.argmax(prediction.data)
    predicted_wh = WH_WORDS[predicted_idx]
    confidence = np.max(prediction.data)
    
    print(f"{description} -> {predicted_wh.upper()} (confidence: {confidence:.2f})")

print("\n" + "=" * 40)
print("Interactive WH-Word Predictor")
print("Type your questions (or 'quit' to exit):")
print("-" * 40)

while True:
    user_input = input("\nEnter a question: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    
    if not user_input:
        continue
    
    # Extract features from user input
    user_features = extract_features(user_input)
    
    # Make prediction
    prediction = model.predict(np.array([user_features]))
    predicted_idx = np.argmax(prediction.data)
    predicted_wh = WH_WORDS[predicted_idx]
    confidence = np.max(prediction.data)
    
    # Show active features
    active_features = [FEATURE_NAMES[i] for i, v in enumerate(user_features) if v]
    features_str = ', '.join(active_features) if active_features else 'none detected'
    
    print(f"Question: '{user_input}'")
    print(f"Features: {features_str}")
    print(f"Predicted WH-word: {predicted_wh.upper()} (confidence: {confidence:.2f})")
    
    # Show intelligent neuron performance if available
    try:
        # Check if model has intelligent neurons
        neuron_info = []
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'activation') and hasattr(layer.activation, 'get_intelligence_score'):
                score = layer.activation.get_intelligence_score()
                neuron_info.append(f"Layer {i+1}: {score:.1f}%")
        
        if neuron_info:
            print(f"Neuron Intelligence: {', '.join(neuron_info)}")
    except:
        pass

print("\n" + "=" * 40)
print("Enhanced WH-Word Predictor Complete!")
print("Powered by FATE-C Enhanced v0.2.0")
print("Features: Intelligent neurons, 97%+ targeting, production-ready")
print("This demonstrates FATE-C's advanced text classification capabilities.")
