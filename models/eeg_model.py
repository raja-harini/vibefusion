import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Path to labels CSV (update as needed)
LABELS_CSV_PATH = 'data/gameemo_train_subset/labels_train.csv'

# Load label CSV to dictionary: filename -> emotion string
label_df = pd.read_csv(LABELS_CSV_PATH)
print("Columns in labels CSV:", label_df.columns)

# Use correct column name as per your CSV
file_to_label = dict(zip(label_df['file_path'], label_df['emotion']))

# Map emotion strings to integer class IDs (update classes & labels per your dataset)
label_map = {'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3, 'fearful': 4}

def load_label_for_file(file_path):
    # Convert Tensor (file path) to string, get relative path matching label CSV key
    path_str = file_path.numpy().decode('utf-8')
    # Match only relative path from your dataset root, adjust as needed
    relative_path = path_str.split('data\\')[-1].replace('\\', '/')
    label_str = file_to_label.get(relative_path, 'neutral')  # default neutral
    label = label_map.get(label_str, 0)
    return label

def parse_fn(file_path):
    # Load EEG segment numpy array from .npy file
    np_path = file_path.numpy().decode('utf-8')
    eeg_data = np.load(np_path).astype(np.float32)  # expected shape: (seq_len, channels)
    
    # Get label integer
    label = load_label_for_file(file_path)
    
    # If 1D data, expand dims for Conv1D input
    if eeg_data.ndim == 1:
        eeg_data = np.expand_dims(eeg_data, axis=-1)
    
    return eeg_data, label

def tf_parse_fn(file_path):
    eeg_data, label = tf.py_function(func=parse_fn, inp=[file_path], Tout=[tf.float32, tf.int32])
    eeg_data.set_shape([256, 1])  # Adjust shape based on your data
    label.set_shape([])
    return eeg_data, label

def get_all_npy_files(data_dir):
    npy_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files

def load_eeg_dataset(data_dir, batch_size=32):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    file_paths = get_all_npy_files(data_dir)
    if len(file_paths) == 0:
        raise ValueError(f"No numpy files found in {data_dir} or its subdirectories")

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(tf_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(file_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # repeat indefinitely for training
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def create_eeg_cnn(input_shape=(256, 1), num_classes=5):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    train_folder = 'data/processed/eeg/train'  # Update with your EEG processed folder

    train_dataset = load_eeg_dataset(train_folder, batch_size=32)

    model = create_eeg_cnn()

    callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('models/eeg_emotion_model.keras', save_best_only=True, monitor='loss')
    ]

    # Steps per epoch based on number of samples and batch size
    steps_per_epoch = max(len(get_all_npy_files(train_folder)) // 32, 1)

    history = model.fit(
        train_dataset,
        epochs=30,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch
    )
