import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.5, highcut=50, fs=256, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def preprocess_eeg_signal(eeg_data, output_dir, sample_name):
    os.makedirs(output_dir, exist_ok=True)
    filtered = bandpass_filter(eeg_data)
    norm_data = (filtered - np.mean(filtered)) / np.std(filtered)
    window_size = 256
    segments = []
    for start in range(0, len(norm_data) - window_size + 1, window_size):
        segment = norm_data[start:start+window_size]
        segments.append(segment)
        np.save(os.path.join(output_dir, f"{sample_name}_seg_{start}.npy"), segment)
    print(f"Saved {len(segments)} EEG segments for {sample_name}")

def preprocess_all_eeg_files(input_base_dir, output_base_dir):
    # Normalize path and escape parentheses for glob
    input_base_dir = os.path.normpath(input_base_dir)
    search_folder = os.path.join(input_base_dir, '**', 'Preprocessed EEG Data', '.csv format', '*.csv')

    # Escape parentheses for Windows glob pattern
    def escape_parentheses(path):
        return path.replace('(', '[()]').replace(')', '[)]')

    escaped_search_folder = escape_parentheses(search_folder)

    file_list = glob.glob(escaped_search_folder, recursive=True)
    print(f"Found {len(file_list)} EEG CSV files in {input_base_dir} for preprocessing.")

    if not file_list:
        print("No files found. Please check folder names & paths carefully.")
        return

    for file_path in file_list:
        try:
            eeg_df = pd.read_csv(file_path)
            eeg_signal = eeg_df.mean(axis=1).values  # average all channels
            relative_path = os.path.relpath(file_path, input_base_dir)
            relative_dir = os.path.dirname(relative_path)
            out_dir = os.path.join(output_base_dir, relative_dir)
            sample_name = os.path.splitext(os.path.basename(file_path))[0]
            preprocess_eeg_signal(eeg_signal, out_dir, sample_name)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    preprocess_all_eeg_files('data/gameemo_train_subset', 'data/processed/eeg/train')
    preprocess_all_eeg_files('data/gameemo_test_subset', 'data/processed/eeg/test')
