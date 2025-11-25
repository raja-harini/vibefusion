import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load saved model
model_path = 'models/eeg_emotion_model.keras'
model = load_model(model_path)

print(f"Loaded model from: {model_path}")
print(model.summary())

# Paths
test_label_csv = 'data/gameemo_test_subset/labels_test.csv'
test_data_dir = 'data/processed/eeg/test'

# Load test CSV with file paths and labels
test_df = pd.read_csv(test_label_csv)
print("Unique labels in test set:", test_df['emotion'].unique())

# Define emotion labels EXACTLY in the order model was trained
emotion_labels = ['neutral', 'joy', 'surprise', 'anger', 'fear', 'sadness', 'disgust']  # Adjust as per training
label_to_id = {label: idx for idx, label in enumerate(emotion_labels)}

print("Emotion labels:", emotion_labels)
print("Test label distribution:\n", test_df['emotion'].value_counts())

# Prepare test features and true labels arrays
X_test = []
y_true = []

for idx, row in test_df.iterrows():
    npy_path = os.path.join(test_data_dir, row['file_path'])
    if os.path.exists(npy_path):
        feature = np.load(npy_path)
        # Debug shape and dtype checks
        if idx < 3:
            print(f"Sample {idx} loaded from {row['file_path']} with shape {feature.shape}")
            print(f"Feature min/max before normalization: {feature.min()}/{feature.max()}")

        # Normalize data, normalize per sample is recommended if needed
        norm_feature = (feature - np.mean(feature)) / np.std(feature)
        
        # Add channel dim if needed
        norm_feature = np.expand_dims(norm_feature, axis=0)

        X_test.append(norm_feature)
        y_true.append(label_to_id.get(row['emotion'], label_to_id['neutral']))
    else:
        print(f"Warning: Feature file not found: {npy_path}")

if not X_test:
    raise Exception("No test features loaded! Check paths and data availability.")

X_test = np.array(X_test)
y_true = np.array(y_true)

print('Loaded X_test shape:', X_test.shape)
print("Sample input shape per example:", X_test[0].shape)
print("Model input shape expected:", model.input_shape)

# Predict on test data
y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs.argmax(axis=1)

print("Unique true labels:", np.unique(y_true))
print("Unique predicted labels:", np.unique(y_pred))

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
