import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Load label mapping from CSV
LABELS_CSV_PATH = 'data/processed/facial/labels_train.csv'  # update path if needed
label_df = pd.read_csv(LABELS_CSV_PATH)
file_to_label = dict(zip(label_df['file_name'], label_df['label']))


def load_label_for_file(file_path):
    file_name = os.path.basename(file_path.numpy().decode('utf-8'))
    label_str = file_to_label.get(file_name, None)
    label_map = {'happy': 0, 'sad': 1, 'neutral': 2, 'angry': 3, 'fearful': 4, 'disgust': 5, 'surprise': 6}
    label = label_map.get(label_str, 2)  # default to neutral if not found
    return label


def parse_fn(file_path):
    img = np.load(file_path.numpy().decode('utf-8'))  # Load numpy image array
    label = load_label_for_file(file_path)

    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    return img, label


def tf_parse_fn(file_path):
    img, label = tf.py_function(parse_fn, [file_path], [tf.float32, tf.int32])
    img.set_shape([48, 48, 1])
    label.set_shape([])
    return img, label


def get_all_npy_files(data_dir):
    npy_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files


def load_facial_dataset(data_dir, batch_size=32):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    file_paths = get_all_npy_files(data_dir)

    if len(file_paths) == 0:
        raise ValueError(f"No numpy files found in {data_dir} or its subdirectories")

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(tf_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(file_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # Repeat dataset indefinitely
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def create_facial_cnn(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    train_folder = 'data/processed/facial/train'  # Adjust if needed
    train_dataset = load_facial_dataset(train_folder, batch_size=32)

    model = create_facial_cnn()

    callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('models/facial_emotion_model.keras', save_best_only=True, monitor='loss')
    ]

    steps_per_epoch = max(len(get_all_npy_files(train_folder)) // 32, 1)

    history = model.fit(
        train_dataset,
        epochs=30,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch
    )
