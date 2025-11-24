import os
import csv
import pandas as pd
import numpy as np
from scipy.signal import welch


def compute_band_power(data, fs, band):
    f, psd = welch(data, fs, nperseg=1024)
    idx_band = (f >= band[0]) & (f <= band[1])
    # Replaced deprecated np.trapz with np.trapezoid
    band_power = np.trapezoid(psd[idx_band], f[idx_band])
    return band_power


def extract_ear_emotion(eeg_df, fs=128):
    # Flatten MultiIndex columns if present
    if isinstance(eeg_df.columns, pd.MultiIndex):
        eeg_df.columns = ['_'.join(map(str, col)).strip() for col in eeg_df.columns.values]

    # Remove unnamed columns if any
    eeg_df = eeg_df.loc[:, ~eeg_df.columns.str.contains('^Unnamed')]

    # Print columns for debugging
    print("Available columns:", eeg_df.columns.tolist())

    # Define required EEG channels
    channel_keys = ["F3", "F4", "AF3", "AF4"]

    def find_channel_prefix(ch_prefix):
        for col in eeg_df.columns:
            if col.startswith(ch_prefix):
                return col
        raise ValueError(f"Missing required EEG channel starting with: {ch_prefix}")

    try:
        channels = {ch: find_channel_prefix(ch) for ch in channel_keys}
    except ValueError as e:
        print(e)
        # Optional: skip this file by returning None
        return None, None

    # Define EEG bands
    alpha_band = (8, 13)
    beta_band = (13, 30)

    # Compute band powers for each channel and band
    F3_alpha = compute_band_power(eeg_df[channels['F3']].values, fs, alpha_band)
    F4_alpha = compute_band_power(eeg_df[channels['F4']].values, fs, alpha_band)
    AF3_alpha = compute_band_power(eeg_df[channels['AF3']].values, fs, alpha_band)
    AF4_alpha = compute_band_power(eeg_df[channels['AF4']].values, fs, alpha_band)

    F3_beta = compute_band_power(eeg_df[channels['F3']].values, fs, beta_band)
    F4_beta = compute_band_power(eeg_df[channels['F4']].values, fs, beta_band)
    AF3_beta = compute_band_power(eeg_df[channels['AF3']].values, fs, beta_band)
    AF4_beta = compute_band_power(eeg_df[channels['AF4']].values, fs, beta_band)

    valence = F4_alpha - F3_alpha
    arousal_num = F3_beta + F4_beta + AF3_beta + AF4_beta
    arousal_den = F3_alpha + F4_alpha + AF3_alpha + AF4_alpha
    arousal = arousal_num / (arousal_den + 1e-8)

    # Normalize values (example scaling)
    valence_norm = np.tanh(valence / 1000)
    arousal_norm = np.tanh((arousal - 1) * 10)

    return valence_norm, arousal_norm


def classify_emotion(valence, arousal):
    if valence is None or arousal is None:
        return 'unknown'
    if valence > 0 and arousal > 0:
        return 'happy/excited'
    elif valence > 0 and arousal <= 0:
        return 'relaxed/content'
    elif valence <= 0 and arousal > 0:
        return 'angry/frustrated'
    else:
        return 'sad/depressed'


def create_eeg_metadata_with_valar(base_dir, output_csv, fs=128):
    metadata_entries = []
    print(f"Processing EEG data in {base_dir}")

    for subject in os.listdir(base_dir):
        subject_path = os.path.join(base_dir, subject)
        if not os.path.isdir(subject_path):
            continue

        preprocessed_path = os.path.join(subject_path, "Preprocessed EEG Data", ".csv format")
        if not os.path.exists(preprocessed_path):
            print(f"Skipping missing folder: {preprocessed_path}")
            continue

        eeg_files = [f for f in os.listdir(preprocessed_path) if f.endswith(".csv")]
        print(f"Found {len(eeg_files)} EEG CSV files for subject {subject}")

        for eeg_file in eeg_files:
            eeg_path = os.path.join(preprocessed_path, eeg_file)
            try:
                # Try reading with multi-level header then fallback to single-level header
                try:
                    eeg_df = pd.read_csv(eeg_path, header=[0, 1])
                    # Flatten multi-index columns if detected
                    if isinstance(eeg_df.columns, pd.MultiIndex):
                        eeg_df.columns = ['_'.join(map(str, col)).strip() for col in eeg_df.columns.values]
                except Exception:
                    eeg_df = pd.read_csv(eeg_path, header=0)

                valence, arousal = extract_ear_emotion(eeg_df, fs=fs)
                emotion = classify_emotion(valence, arousal)
                print(f"File: {eeg_file} Valence: {valence} Arousal: {arousal} Emotion: {emotion}")

            except Exception as e:
                print(f"Error processing {eeg_file}: {e}")
                emotion = 'unknown'

            relative_path = os.path.relpath(eeg_path, base_dir)
            metadata_entries.append((relative_path, emotion))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_path', 'emotion'])
        writer.writerows(metadata_entries)

    print(f"Metadata with Valence/Arousal emotions created at: {output_csv}")


if __name__ == "__main__":
    eeg_base_dir = "data/gameemo_train_subset/"  # Adjust as necessary
    output_csv_path = "data/gameemo_train_subset/eeg_metadata_train.csv"
    create_eeg_metadata_with_valar(eeg_base_dir, output_csv_path)
