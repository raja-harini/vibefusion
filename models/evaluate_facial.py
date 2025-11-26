import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils import class_weight


# Load saved model
model_path = 'models/facial_emotion_model.keras'
model = load_model(model_path)
print(f"Loaded model from: {model_path}")
print(model.summary())


# Paths - adjust according to your repo structure
test_label_csv = 'data/processed/facial/labels_test.csv'
test_data_dir = 'data/processed/facial/test'


# Load test CSV with file names and labels
test_df = pd.read_csv(test_label_csv)
print("Unique labels in test set:", test_df['label'].unique())


# Define emotion labels and mapping (ensure consistent with training)
emotion_labels = ['neutral', 'fear', 'disgust', 'sadness', 'surprise', 'anger', 'joy']
label_to_id = {label: idx for idx, label in enumerate(emotion_labels)}


# Filter valid labels only
test_df = test_df[test_df['label'].isin(emotion_labels)]


# Prepare test features and true labels
X_test = []
y_true = []


for idx, row in test_df.iterrows():
    npy_path = os.path.join(test_data_dir, row['file_name'])
    if os.path.exists(npy_path):
        feature = np.load(npy_path).astype('float32')
        # Normalize pixel values
        norm_feature = feature / 255.0
        # Add channel dimension if missing (48x48 -> 48x48x1)
        if norm_feature.ndim == 2:
            norm_feature = np.expand_dims(norm_feature, axis=-1)
        X_test.append(norm_feature)
        y_true.append(label_to_id[row['label']])
    else:
        print(f"Warning: Feature file not found: {npy_path}")


if not X_test:
    raise Exception("No test features loaded! Check paths and data availability.")


X_test = np.array(X_test)
y_true = np.array(y_true)


print("Loaded X_test shape:", X_test.shape)
print("Sample input shape per example:", X_test[0].shape)
print("Model input shape expected:", model.input_shape)


# Predict on test data
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)


print("Unique true labels in test:", np.unique(y_true))
print("Unique predicted labels:", np.unique(y_pred))


# Classification metrics
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(emotion_labels))))
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=list(range(len(emotion_labels))), average='weighted', zero_division=0
)


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


# Plot confusion matrix heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_labels,
            yticklabels=emotion_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Facial Emotion Recognition')
plt.show()


# ---- Training snippet starts here ----
# NOTE: You must provide/load the training data arrays for this part to run successfully.

# Example placeholders: replace these with actual loaded training datasets
# X_train = np.load('path_to_preprocessed_train_features.npy')
# y_train = np.load('path_to_train_labels.npy')
# val_dataset = (X_val, y_val) or create tf.data.Dataset for validation

# For demonstration, create dummy data with correct shape to avoid errors
# Remove this block when actual train data is available
if 'X_train' not in globals():
    X_train = np.random.rand(100, 48, 48, 1).astype('float32')
if 'y_train' not in globals():
    y_train = np.random.randint(0, len(emotion_labels), size=(100,))
if 'val_dataset' not in globals():
    val_dataset = (np.random.rand(20, 48, 48, 1).astype('float32'), np.random.randint(0, len(emotion_labels), size=(20,)))

# Compute balanced class weights to address imbalance
class_weights_array = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(y_train),
                                                      y=y_train)
class_weights_dict = dict(enumerate(class_weights_array))
print("Class weights:", class_weights_dict)


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(model_path, save_best_only=True)


history = model.fit(X_train, y_train,
                    validation_data=val_dataset,
                    epochs=30,
                    class_weight=class_weights_dict,
                    callbacks=[early_stop, checkpoint])


# Plot training history (loss and accuracy)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()


plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()
