import os
import csv
import pandas as pd

# Paths
facial_data_dir = 'data/processed/facial/test'
original_labels_csv = 'data/meld_test_subset/test_sent_emo.csv'  # adjust path as needed
output_csv = 'data/processed/facial/labels_test.csv'

# Load original labels CSV into a dataframe
df_labels = pd.read_csv(original_labels_csv)

# Create a mapping from utterance to emotion (e.g., 'dia0_utt0' -> 'happy')
def utterance_code(row):
    # Construct keys matching folder names, adjust according to naming
    return f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"

df_labels['folder_name'] = df_labels.apply(utterance_code, axis=1)
utterance_to_emotion = dict(zip(df_labels['folder_name'], df_labels['Emotion']))

# Prepare to write new labels CSV for npy files
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['file_name', 'label'])

    # Iterate through all utterance folders
    for utterance_folder in os.listdir(facial_data_dir):
        folder_path = os.path.join(facial_data_dir, utterance_folder)
        if os.path.isdir(folder_path):
            emotion_label = utterance_to_emotion.get(utterance_folder, 'neutral')  # default to neutral if not found
            for npy_file in os.listdir(folder_path):
                if npy_file.endswith('.npy'):
                    writer.writerow([f"{utterance_folder}/{npy_file}", emotion_label])

print(f"Generated label file for facial training: {output_csv}")
