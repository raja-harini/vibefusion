import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd

# Load label mapping csv once globally
LABELS_CSV_PATH = 'processed/facial/labels_train.csv'  # Update path as per your structure
label_df = pd.read_csv(LABELS_CSV_PATH)
file_to_label = dict(zip(label_df['file_name'], label_df['label']))  # Adjust column names accordingly

def load_label_for_file(file_path):
    file_name = os.path.basename(file_path.numpy().decode('utf-8'))
    label_str = file_to_label.get(file_name, None)
    # Map label_str to integer class index
    label_map = {'happy': 0, 'sad': 1, 'neutral': 2, 'angry': 3, 'fearful': 4, 'disgust': 5, 'surprise': 6}
    label = label_map.get(label_str, 2)  # default to neutral if unknown
    return label

def parse_fn(file_path):
    img = np.load(file_path.numpy().decode('utf-8'))  # Load numpy array image
    label = load_label_for_file(file_path)

    # Normalize image pixels to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Ensure image shape is (48,48,1)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    return img, label

def tf_parse_fn(file_path):
    img, label = tf.py_function(parse_fn, [file_path], [tf.float32, tf.int32])
    img.set_shape([48, 48, 1])
    label.set_shape([])
    return img, label

def load_facial_dataset(data_dir, batch_size=32):
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(tf_parse_fn)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def create_facial_cnn(input_shape=(48,48,1), num_classes=7):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Update with your actual data folders
    train_folder = 'processed/facial/train'
    val_folder = 'processed/facial/val'

    train_dataset = load_facial_dataset(train_folder)
    val_dataset = load_facial_dataset(val_folder)

    model = create_facial_cnn()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('models/facial_emotion_model.h5', save_best_only=True)
    ]

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=30, callbacks=callbacks)
