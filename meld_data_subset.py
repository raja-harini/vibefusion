import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_subset(input_csv_path, output_train_csv, output_test_csv, sample_per_class=10, test_size=0.3):
    df = pd.read_csv(input_csv_path)
    # Balance classes by sampling equal examples per emotion
    sampled = df.groupby('Emotion', group_keys=False).apply(lambda x: x.sample(min(len(x), sample_per_class)))
    # Split into train/test
    train_df, test_df = train_test_split(sampled, test_size=test_size, stratify=sampled['Emotion'], random_state=42)
    # Save subset CSVs
    os.makedirs(os.path.dirname(output_train_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_csv), exist_ok=True)
    train_df.to_csv(output_train_csv, index=False)
    test_df.to_csv(output_test_csv, index=False)
    print(f"Subsets created from {input_csv_path} with {len(train_df)} train and {len(test_df)} test samples.")

# Paths - Modify these to your actual MELD CSV locations
meld_train_csv = 'data/meld_train_subset/train_sent_emo.csv'
meld_test_csv = 'data/meld_test_subset/test_sent_emo.csv'  # if you have a test CSV, else combine with train CSV

# Create subsets for train CSV
create_subset(meld_train_csv, 'data/meld_train_subset/labels_train.csv', 'data/meld_test_subset/labels_test.csv')

# If you have a separate test CSV, create subsets too
# create_subset(meld_test_csv, 'data/meld_test_subset/labels_train.csv', 'data/meld_test_subset/labels_test.csv')
