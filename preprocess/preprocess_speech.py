import os
import subprocess
import librosa
import numpy as np

def extract_audio_features(video_path, output_dir, sr=22050):
    os.makedirs(output_dir, exist_ok=True)
    
    wav_filename = os.path.join(output_dir, os.path.basename(video_path).replace('.mp4', '.wav'))

    # Use ffmpeg subprocess to extract audio ignoring chapters and subtitles
    cmd = ['ffmpeg', '-y', '-i', video_path,
           '-map', '0:a:0',        # map only the first audio stream
           '-map_chapters', '-1',  # ignore chapters metadata
           '-vn',                  # disable video stream
           wav_filename]
    
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    
    # Load audio and compute MFCC features
    y, sr = librosa.load(wav_filename, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    # Save features as .npy file
    feat_filename = os.path.join(output_dir, os.path.basename(video_path).replace('.mp4', '_mfcc.npy'))
    np.save(feat_filename, mfccs_mean)
    
    # Remove temporary audio file after processing
    os.remove(wav_filename)
    print(f"Processed audio features for {video_path}")

def preprocess_all_videos(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_dir, filename)
            try:
                extract_audio_features(video_path, output_dir)
            except subprocess.CalledProcessError as e:
                print(f"Error processing {video_path}: {e}")

# Usage: preprocess all videos in meld_train_subset folder
if __name__ == "__main__":
    preprocess_all_videos('data/meld_train_subset', 'data/processed/speech/train')
    preprocess_all_videos('data/meld_test_subset', 'data/processed/speech/test')
