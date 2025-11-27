import sys
import os
import time
import socketio
import cv2
from datetime import datetime

# Add current directory to system path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.facial_emotion import get_facial_emotion
from modules.speech_emotion import get_speech_emotion
from modules.eeg_emotion import get_eeg_emotion
from fusion import fuse_emotions

# Import alert system components
from modules.alert_system import emotion_history, check_alerts

# Initialize SocketIO client
sio = socketio.Client()

@sio.event
def connect():
    print("Connected to SocketIO server")

@sio.event
def disconnect():
    print("Disconnected from SocketIO server")

def send_emotion_to_web(emotion):
    if emotion and sio.connected:
        sio.emit('send_emotion', {'emotion': emotion})
    else:
        print("SocketIO not connected. Cannot send emotion.")

def main_loop():
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            f_emotion = get_facial_emotion(frame)
            s_emotion = get_speech_emotion()
            e_emotion = get_eeg_emotion()

            # Debug prints to verify types and values
            print("Facial emotion type:", type(f_emotion), "value:", f_emotion)
            print("Speech emotion type:", type(s_emotion), "value:", s_emotion)
            print("EEG emotion type:", type(e_emotion), "value:", e_emotion)

            combined_emotion = fuse_emotions(f_emotion, s_emotion, e_emotion)

            print("Facial:", f_emotion, "| Speech:", s_emotion, "| EEG:", e_emotion)
            print("Predicted Emotion:", combined_emotion)

            # Append latest combined emotion to history buffer
            emotion_history.append(combined_emotion)

            # Check for alert and trigger if needed
            if check_alerts(combined_emotion):
                print("Alert triggered due to emotional fluctuation.")

            # Send combined emotion to web app via SocketIO
            send_emotion_to_web(combined_emotion)

            time.sleep(1)
    finally:
        cap.release()

def main():
    try:
        # Connect to SocketIO server (default namespace '/')
        sio.connect('http://localhost:5000', namespaces=['/'])
        main_loop()
    except KeyboardInterrupt:
        print("Exiting emotion detection.")
    finally:
        if sio.connected:
            sio.disconnect()

if __name__ == "__main__":
    main()
