import os
import random
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load label mapping from CSV
LABELS_CSV_PATH = 'data/processed/facial/labels_train.csv'  # update path if required
label_df = pd.read_csv(LABELS_CSV_PATH)
file_to_label = dict(zip(label_df['file_name'], label_df['label']))

emotion_labels = ['neutral', 'fear', 'disgust', 'sadness', 'surprise', 'anger', 'joy']
label_map = {label: idx for idx, label in enumerate(emotion_labels)}

def balance_dataset(label_csv, data_dir):
    df = pd.read_csv(label_csv)
    emotion_counts = df['label'].value_counts()
    min_count = emotion_counts.min()
    balanced_files = []

    for emotion in emotion_counts.index:
        emotion_files = df[df['label'] == emotion]['file_name'].tolist()
        if len(emotion_files) > min_count:
            emotion_files = random.sample(emotion_files, min_count)
        balanced_files.extend([(os.path.join(data_dir, f), emotion) for f in emotion_files])

    random.shuffle(balanced_files)
    return balanced_files

def load_label_for_file(file_path):
    file_name = os.path.basename(file_path.numpy().decode('utf-8'))
    label_str = file_to_label.get(file_name, 'neutral')
    label = label_map.get(label_str, label_map['neutral'])
    return label

def parse_fn(file_path):
    img = np.load(file_path.numpy().decode('utf-8'))  # Load numpy image array
    norm_img = img.astype(np.float32) / 255.0  # Normalize to [0,1]

    if norm_img.ndim == 2:
        norm_img = np.expand_dims(norm_img, axis=-1)

    label = load_label_for_file(file_path)
    return norm_img, label

def tf_parse_fn(file_path):
    img, label = tf.py_function(parse_fn, [file_path], [tf.float32, tf.int32])
    img.set_shape([48, 48, 1])
    label.set_shape([])
    return img, label

def load_facial_dataset(data_dir, batch_size=32, shuffle=True, repeat=True):
    npy_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))

    dataset = tf.data.Dataset.from_tensor_slices(npy_files)
    dataset = dataset.map(tf_parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(npy_files))
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def load_facial_dataset_balanced(balanced_files, batch_size=32, shuffle=True, repeat=True):
    file_paths = [f[0] for f in balanced_files]
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(tf_parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_paths))
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def get_all_npy_files(data_dir):
    npy_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files

def create_facial_cnn(input_shape=(48, 48, 1), num_classes=len(emotion_labels)):
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
    train_folder = 'data/processed/facial/train'
    val_folder = 'data/processed/facial/val'

    # Create balanced training file list with labels
    balanced_files = balance_dataset(LABELS_CSV_PATH, train_folder)
    train_dataset = load_facial_dataset_balanced(balanced_files, batch_size=32, shuffle=True, repeat=True)
    val_dataset = load_facial_dataset(val_folder, batch_size=32, shuffle=False, repeat=False)

    steps_per_epoch = max(len(balanced_files) // 32, 1)
    validation_steps = max(len(get_all_npy_files(val_folder)) // 32, 1)

    model = create_facial_cnn()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('models/facial_emotion_model.keras', save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(
        train_dataset,
        epochs=30,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps
    )
