import sys
import os
import time
import socketio

# Add current directory to system path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.facial_emotion import get_facial_emotion
from modules.speech_emotion import get_speech_emotion
from modules.eeg_emotion import get_eeg_emotion
from fusion import fuse_emotions

# Connect to Flask SocketIO server
sio = socketio.Client()
sio.connect('http://localhost:5000')  # Change if deployed elsewhere

def send_emotion_to_web(emotion):
    if emotion:
        sio.emit('send_emotion', {'emotion': emotion})

def main():
    try:
        while True:
            f_emotion = get_facial_emotion()
            s_emotion = get_speech_emotion()
            e_emotion = get_eeg_emotion()

            # Debug prints to diagnose type issues and values
            print("Facial emotion type:", type(f_emotion), "value:", f_emotion)
            print("Speech emotion type:", type(s_emotion), "value:", s_emotion)
            print("EEG emotion type:", type(e_emotion), "value:", e_emotion)

            combined_emotion = fuse_emotions(f_emotion, s_emotion, e_emotion)

            print("Facial:", f_emotion, "| Speech:", s_emotion, "| EEG:", e_emotion)
            print("Predicted Emotion:", combined_emotion)

            # Send combined emotion to Flask web app
            send_emotion_to_web(combined_emotion)

            # Sleep briefly to avoid spamming updates; adjust as needed
            time.sleep(1)

    except KeyboardInterrupt:
        print("Exiting emotion detection.")

if __name__ == "__main__":
    main()
