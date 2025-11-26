import os
import csv
import pandas as pd

# Paths - adjust if changed
eeg_data_dir = 'data/processed/eeg/test'  # folder structure containing .npy or .csv EEG files
metadata_csv_path = 'data/gameemo_test_subset/eeg_metadata_test.csv'  # your metadata CSV with labels
output_csv_path = 'data/processed/eeg/labels_test.csv'  # output CSV mapping filenames to labels

# Load EEG metadata CSV into DataFrame.
# This CSV must at minimum have columns: 'file_name' (exact file name or relative path) and 'emotion'
df_metadata = pd.read_csv(metadata_csv_path)

# Build a dictionary mapping file names to emotions
metadata_mapping = dict(zip(df_metadata['file_path'], df_metadata['emotion']))

# Write mapping CSV for files present in eeg_data_dir
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['file_path', 'label'])
    
    for root, dirs, files in os.walk(eeg_data_dir):
        for file_name in files:
            if file_name.endswith('.npy') or file_name.endswith('.csv'):
                # Construct relative path from eeg_data_dir for consistency
                rel_dir = os.path.relpath(root, eeg_data_dir)
                rel_path = os.path.join(rel_dir, file_name) if rel_dir != '.' else file_name
                
                # Lookup emotion label; default 'neutral' if missing
                emotion = metadata_mapping.get(rel_path, 'neutral')
                
                writer.writerow([rel_path, emotion])

print(f"Generated EEG label CSV at: {output_csv_path}")
