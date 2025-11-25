import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Load label mapping from CSV
LABELS_CSV_PATH = 'data/processed/speech/labels_train.csv'  # Update if different
label_df = pd.read_csv(LABELS_CSV_PATH)
file_to_label = dict(zip(label_df['file_name'], label_df['label']))

# Map emotion strings to integer class indices
label_map = {'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3, 'fearful': 4}


def load_label_for_file(file_path):
    file_name = os.path.basename(file_path.numpy().decode('utf-8'))
    label_str = file_to_label.get(file_name, 'neutral')  # default neutral if not found
    return label_map.get(label_str, 0)


def parse_fn(file_path):
    # Load preprocessed MFCC numpy array
    features = np.load(file_path.numpy().decode('utf-8')).astype(np.float32)
    # Features expected shape: (n_mfcc,), expand dims to (timesteps, feature_dim) if needed
    # For simplicity, reshape as (timesteps=1, feature_dim=n_mfcc)
    features = np.expand_dims(features, axis=0)
    
    label = load_label_for_file(file_path)
    return features, label


def tf_parse_fn(file_path):
    features, label = tf.py_function(parse_fn, [file_path], [tf.float32, tf.int32])
    features.set_shape([1, 13])  # 1 timestep, 13 MFCC features
    label.set_shape([])
    return features, label


def get_all_npy_files(data_dir):
    npy_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files


def load_speech_dataset(data_dir, batch_size=32):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    file_paths = get_all_npy_files(data_dir)

    if len(file_paths) == 0:
        raise ValueError(f"No numpy files found in {data_dir} or its subdirectories")

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(tf_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(file_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # Repeat indefinitely
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def create_speech_lstm(input_shape=(1, 13), num_classes=5):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(64),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    train_folder = 'data/processed/speech/train'  # Adjust if needed
    train_dataset = load_speech_dataset(train_folder, batch_size=32)

    model = create_speech_lstm()

    callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('models/speech_emotion_model.keras', save_best_only=True, monitor='loss')
    ]

    steps_per_epoch = max(len(get_all_npy_files(train_folder)) // 32, 1)

    history = model.fit(
        train_dataset,
        epochs=30,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch
    )
