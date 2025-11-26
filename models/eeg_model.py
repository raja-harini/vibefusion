import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Load label mapping from CSV
LABELS_CSV_PATH = 'data/processed/eeg/labels_train.csv'  # update path if needed
label_df = pd.read_csv(LABELS_CSV_PATH)
file_to_label = dict(zip(label_df['file_path'], label_df['label']))


emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful']
label_map = {label: idx for idx, label in enumerate(emotion_labels)}


def load_label_for_file(file_path):
    # Convert Tensor to string path and get relative path matching label CSV keys
    path_str = file_path.numpy().decode('utf-8')
    relative_path = path_str.split('data\\')[-1].replace('\\', '/')
    label_str = file_to_label.get(relative_path, 'neutral')  # default neutral
    label = label_map.get(label_str, label_map['neutral'])
    return label


def parse_fn(file_path):
    np_path = file_path.numpy().decode('utf-8')
    eeg_data = np.load(np_path).astype(np.float32)  # expected shape: (seq_len, channels)

    label = load_label_for_file(file_path)

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
    dataset = dataset.map(tf_parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(file_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # repeat indefinitely for training
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_eeg_cnn(input_shape=(256, 1), num_classes=len(emotion_labels)):
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
    train_folder = 'data/processed/eeg/train'  # Adjust paths as needed
    val_folder = 'data/processed/eeg/val'      # Optionally create validation folder

    train_dataset = load_eeg_dataset(train_folder, batch_size=32)
    val_dataset = load_eeg_dataset(val_folder, batch_size=32) if os.path.exists(val_folder) else None

    steps_per_epoch = max(len(get_all_npy_files(train_folder)) // 32, 1)
    validation_steps = max(len(get_all_npy_files(val_folder)) // 32, 1) if val_dataset else None

    model = create_eeg_cnn()

    callbacks = [
        EarlyStopping(monitor='val_loss' if val_dataset else 'loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('models/eeg_emotion_model.keras', save_best_only=True,
                        monitor='val_loss' if val_dataset else 'loss')
    ]

    history = model.fit(
        train_dataset,
        epochs=30,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps
    )
