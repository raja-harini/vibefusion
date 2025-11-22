import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.facial_emotion import get_facial_emotion
from modules.speech_emotion import get_speech_emotion
from modules.eeg_emotion import get_eeg_emotion
from fusion import fuse_emotions

while True:
    f_emotion = get_facial_emotion()
    s_emotion = get_speech_emotion()
    e_emotion = get_eeg_emotion()

    # Debug prints to diagnose type issues
    print("Facial emotion type:", type(f_emotion), "value:", f_emotion)
    print("Speech emotion type:", type(s_emotion), "value:", s_emotion)
    print("EEG emotion type:", type(e_emotion), "value:", e_emotion)

    combined_emotion = fuse_emotions(f_emotion, s_emotion, e_emotion)

    print("Facial:", f_emotion, "| Speech:", s_emotion, "| EEG:", e_emotion)
    print("Predicted Emotion:", combined_emotion)

    # Add delay or exit conditions as needed
