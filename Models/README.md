# Sentiment Analysis Model (Enhanced for >80% Accuracy)

## Model Information
- **Model Type**: Enhanced CNN with BatchNorm + Multi-kernel Architecture
- **Framework**: TensorFlow/Keras
- **Input**: Tokenized text (max_length=100, vocab_size=5000)
- **Output**: 3 classes (negative, neutral, positive)
- **Labeling**: VADER Sentiment Analysis

## Files
- `model_sentiment_cnn.keras`: Trained model
- `tokenizer.pkl`: Text tokenizer (pickle format)
- `label_encoding.npy`: Label encoding mapping

## Usage

### Load Model and Make Predictions
```python
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model('model_sentiment_cnn.keras')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load label classes
label_classes = np.load('label_encoding.npy')

# Prepare text
text = "This movie is amazing!"
sequences = tokenizer.texts_to_sequences([text])
padded = pad_sequences(sequences, maxlen=100)

# Predict
prediction = model.predict(padded)
predicted_class = np.argmax(prediction)
sentiment = label_classes[predicted_class]

print(f"Sentiment: {sentiment}")
print(f"Confidence: {prediction[0][predicted_class]:.2f}")
```

## Enhanced Model Architecture
- **Embedding Layer**: vocab_size=5000, embedding_dim=128
- **Multi-kernel Conv1D Layers**:
  - Conv1D (128 filters, kernel_size=2) + BatchNorm + GlobalMaxPooling
  - Conv1D (128 filters, kernel_size=3) + BatchNorm + GlobalMaxPooling
  - Conv1D (128 filters, kernel_size=4) + BatchNorm + GlobalMaxPooling
  - Conv1D (128 filters, kernel_size=5) + BatchNorm + GlobalMaxPooling
- **Concatenation** of all conv outputs
- **Dense Layers**:
  - Dense (256 units, ReLU) + BatchNorm + Dropout(0.5)
  - Dense (128 units, ReLU) + BatchNorm + Dropout(0.4)
  - Dense (64 units, ReLU) + Dropout(0.3)
- **Output Layer**: 3 units, Softmax

## Training Enhancements
- L2 Regularization (0.001) on Conv and Dense layers
- Learning Rate Scheduling (ReduceLROnPlateau)
- Class Weight Balancing for imbalanced data
- Up to 50 epochs with early stopping (patience=10)

## Target Performance
- **Target Accuracy**: >80%
- **Classes**: Negative (0), Neutral (1), Positive (2)
- **Labeling Method**: VADER Sentiment Analysis
