import sounddevice as sd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import random

# Parameters for recording
duration = 3  # seconds
fs = 22050   # sample rate, Hz

def record_audio(duration=duration, fs=fs):
    print("Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    audio = audio.flatten()
    print("Recording complete.")
    return audio

def extract_features(audio, fs=fs):
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=13)
    # Average MFCCs over time axis
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

# Placeholder for your trained classification model
# Replace this with actual model loading and prediction code
def predict_emotion(features):
    emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful']
    # Randomly pick an emotion as demo (replace with actual model.predict)
    predicted_emotion = random.choice(emotions)
    return predicted_emotion

def get_speech_emotion():
    audio = record_audio()
    features = extract_features(audio)
    # Optional: scale features here if model requires it
    emotion = predict_emotion(features)
    return emotion

# Check function works standalone
if __name__ == "__main__":
    emotion = get_speech_emotion()
    print("Predicted Speech Emotion:", emotion)
