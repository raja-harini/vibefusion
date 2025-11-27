from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from modules.facial_emotion import get_facial_emotion
from modules.speech_emotion import get_speech_emotion
from modules.eeg_emotion import get_eeg_emotion
from fusion import fuse_emotions
import cv2
import threading
import base64
import numpy as np
from collections import deque
import time

# Import alert system components exactly as in vibefusion_main.py
from modules.alert_system import emotion_history, check_alerts, alert_user_and_caregiver

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize webcam with thread lock for safe concurrent access
cap = cv2.VideoCapture(0)
camera_lock = threading.Lock()

def capture_frame():
    with camera_lock:
        ret, frame = cap.read()
    if not ret:
        return None
    return frame

def encode_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

def emit_status(message):
    socketio.emit('backend_status', {'message': message})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-facial-emotion', methods=['GET'])
def facial_emotion_api():
    emit_status("Capturing frame for facial emotion detection...")
    frame = capture_frame()
    if frame is None:
        return jsonify({'error': 'Camera error'}), 500

    emit_status("Running facial emotion detection...")
    facial_emotion = get_facial_emotion(frame)
    emit_status(f"Facial emotion detected: {facial_emotion}")

    # Append to emotion history and check alerts (NEW: Alert system integration)
    if facial_emotion:
        emotion_history.append(facial_emotion)
        if check_alerts(emotion_history):
            alert_message = f"Facial emotion alert: High fluctuation detected - {facial_emotion}"
            alert_user_and_caregiver(alert_message)
            emit_status("🚨 ALERT: Facial emotion fluctuation detected!")

    # Encode frame as base64 to send to client
    frame_encoded = encode_frame_to_base64(frame)

    socketio.emit('update_emotion', {'emotion': facial_emotion, 'image': frame_encoded})
    return jsonify({'facial_emotion': facial_emotion, 'frame': frame_encoded})

@app.route('/get-speech-emotion', methods=['GET'])
def speech_emotion_api():
    emit_status("Starting speech recording and emotion detection...")
    speech_emotion = get_speech_emotion()
    emit_status(f"Speech emotion detected: {speech_emotion}")

    # Append to emotion history and check alerts (NEW: Alert system integration)
    if speech_emotion:
        emotion_history.append(speech_emotion)
        if check_alerts(emotion_history):
            alert_message = f"Speech emotion alert: High fluctuation detected - {speech_emotion}"
            alert_user_and_caregiver(alert_message)
            emit_status("🚨 ALERT: Speech emotion fluctuation detected!")

    socketio.emit('update_emotion', {'emotion': speech_emotion})
    return jsonify({'speech_emotion': speech_emotion})

@app.route('/get-eeg-emotion', methods=['GET'])
def eeg_emotion_api():
    emit_status("Running EEG emotion detection...")
    eeg_emotion = get_eeg_emotion()
    emit_status(f"EEG emotion detected: {eeg_emotion}")

    # Append to emotion history and check alerts (NEW: Alert system integration)
    if eeg_emotion:
        emotion_history.append(eeg_emotion)
        if check_alerts(emotion_history):
            alert_message = f"EEG emotion alert: High fluctuation detected - {eeg_emotion}"
            alert_user_and_caregiver(alert_message)
            emit_status("🚨 ALERT: EEG emotion fluctuation detected!")

    socketio.emit('update_emotion', {'emotion': eeg_emotion})
    return jsonify({'eeg_emotion': eeg_emotion})

@app.route('/get-combined-emotion', methods=['GET'])
def combined_emotion_api():
    emit_status("Capturing frame for combined emotion detection...")
    frame = capture_frame()
    if frame is None:
        return jsonify({'error': 'Camera error'}), 500

    emit_status("Running facial emotion detection for combined...")
    facial_emotion = get_facial_emotion(frame)
    emit_status("Running speech emotion detection for combined...")
    speech_emotion = get_speech_emotion()
    emit_status("Running EEG emotion detection for combined...")
    eeg_emotion = get_eeg_emotion()

    combined_emotion = fuse_emotions(facial_emotion, speech_emotion, eeg_emotion)
    emit_status(f"Combined emotion detected: {combined_emotion}")

    # Append COMBINED emotion to history and check alerts (NEW: Alert system integration)
    if combined_emotion:
        emotion_history.append(combined_emotion)
        if check_alerts(emotion_history):
            alert_message = f"Combined emotion alert: High fluctuation detected - {list(emotion_history)}"
            alert_user_and_caregiver(alert_message)
            emit_status("🚨 ALERT: Combined emotion fluctuation detected!")

    frame_encoded = encode_frame_to_base64(frame)
    socketio.emit('update_emotion', {'emotion': combined_emotion, 'image': frame_encoded})
    return jsonify({
        'facial_emotion': facial_emotion,
        'speech_emotion': speech_emotion,
        'eeg_emotion': eeg_emotion,
        'combined_emotion': combined_emotion,
        'frame': frame_encoded
    })

@socketio.on('send_emotion')
def handle_emotion(data):
    emotion = data.get('emotion')
    # Append received emotion to history and check alerts (NEW: Alert system integration)
    if emotion:
        emotion_history.append(emotion)
        if check_alerts(emotion_history):
            alert_message = f"SocketIO emotion alert: High fluctuation detected - {emotion}"
            alert_user_and_caregiver(alert_message)
            emit('backend_status', {'message': "🚨 ALERT: SocketIO emotion fluctuation detected!"}, broadcast=True)
    
    emit('update_emotion', data, broadcast=True)

@app.route('/shutdown', methods=['POST'])
def shutdown():
    cap.release()
    return 'Server shutting down...'

if __name__ == '__main__':
    socketio.run(app, debug=True)
