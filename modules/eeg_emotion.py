import random
from tensorflow.keras.models import load_model
import numpy as np

# Load trained EEG emotion model once during module import
model = load_model('models/eeg_emotion_model.keras')
emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful']

# Example EEG mock emotion data simulation list
eeg_emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise']

def preprocess_eeg(raw_eeg_segment):
    """
    Normalize and reshape EEG data for model input
    Input:
        raw_eeg_segment - numpy array of EEG features/signals
    Output:
        input_data - preprocessed, reshaped EEG data ready for model prediction
    """
    norm_data = (raw_eeg_segment - np.mean(raw_eeg_segment)) / np.std(raw_eeg_segment)  # normalize
    input_data = np.expand_dims(norm_data, axis=0)  # add batch dimension
    input_data = np.expand_dims(input_data, axis=-1)  # add channel dimension if model expects it
    return input_data

def decode_prediction(preds):
    """
    Decode model output to emotion label string
    """
    return emotion_labels[np.argmax(preds)]

def predict_emotion(raw_eeg_segment):
    """
    Predict emotion from raw EEG segment using the trained model
    """
    input_data = preprocess_eeg(raw_eeg_segment)
    preds = model.predict(input_data)
    emotion = decode_prediction(preds)
    return emotion

def get_eeg_emotion():
    """
    Simulates reading EEG mock data and returns a predicted emotion.
    For now, this randomly chooses an emotion from a predefined list.
    Replace with actual EEG model inference when available.
    """
    # Uncomment and modify below to replace with real EEG data read and model prediction
    # raw_eeg_data = read_your_real_eeg_data()
    # return predict_emotion(raw_eeg_data)

    # For simulation, randomly pick an emotion
    predicted_emotion = random.choice(eeg_emotions)
    return predicted_emotion

# Test functionality standalone
if __name__ == "__main__":
    # Test simulated emotion
    emotion = get_eeg_emotion()
    print(f"Simulated EEG emotion: {emotion}")

    # To test actual model prediction, uncomment and supply real EEG sample:
    # sample_data = np.random.randn(256)  # example dummy EEG segment
    # emotion = predict_emotion(sample_data)
    # print(f"Model predicted EEG emotion: {emotion}")
