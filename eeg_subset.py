import pandas as pd
import os
from sklearn.model_selection import train_test_split

def create_eeg_subset(metadata_csv_path, train_output_csv, test_output_csv, samples_per_class=10, test_size=0.3):
    # Load full EEG metadata CSV with columns at least: file_path, emotion
    df = pd.read_csv(metadata_csv_path)

    # Sample balanced subset: limit samples per emotion to avoid class imbalance
    sampled = df.groupby('emotion', group_keys=False).apply(lambda x: x.sample(n=min(len(x), samples_per_class), random_state=42))

    # Split into train/test with stratification on emotion
    train_df, test_df = train_test_split(sampled, test_size=test_size, stratify=sampled['emotion'], random_state=42)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(train_output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(test_output_csv), exist_ok=True)

    # Save subset CSVs
    train_df.to_csv(train_output_csv, index=False)
    test_df.to_csv(test_output_csv, index=False)

    print(f"EEG subsets created: {len(train_df)} train samples, {len(test_df)} test samples")

if __name__ == "__main__":
    # Define paths (change if necessary)
    metadata_csv = r"C:\Users\LENOVO\vibefusion\data\gameemo_train_subset\eeg_metadata_train.csv"
    train_csv = r"C:\Users\LENOVO\vibefusion\data\gameemo_train_subset\labels_train.csv"
    test_csv = r"C:\Users\LENOVO\vibefusion\data\gameemo_test_subset\labels_test.csv"

    # Create balanced EEG train/test subsets
    create_eeg_subset(metadata_csv, train_csv, test_csv)
