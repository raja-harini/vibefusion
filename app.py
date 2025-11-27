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

# Import alert system components - CORRECTED IMPORT
from modules.alert_system import check_alerts  # Only import function, not shared state

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize webcam with thread lock for safe concurrent access
cap = cv2.VideoCapture(0)
camera_lock = threading.Lock()

# Local emotion history for this Flask app (separate from vibefusion_main.py)
emotion_history = deque(maxlen=30)

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

    # Append to emotion history and check alerts (CORRECTED)
    if facial_emotion:
        emotion_history.append(facial_emotion)
        if check_alerts(facial_emotion):  # Pass emotion directly to check_alerts
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

    # Append to emotion history and check alerts (CORRECTED)
    if speech_emotion:
        emotion_history.append(speech_emotion)
        if check_alerts(speech_emotion):  # Pass emotion directly to check_alerts
            emit_status("🚨 ALERT: Speech emotion fluctuation detected!")

    socketio.emit('update_emotion', {'emotion': speech_emotion})
    return jsonify({'speech_emotion': speech_emotion})

@app.route('/get-eeg-emotion', methods=['GET'])
def eeg_emotion_api():
    emit_status("Running EEG emotion detection...")
    eeg_emotion = get_eeg_emotion()
    emit_status(f"EEG emotion detected: {eeg_emotion}")

    # Append to emotion history and check alerts (CORRECTED)
    if eeg_emotion:
        emotion_history.append(eeg_emotion)
        if check_alerts(eeg_emotion):  # Pass emotion directly to check_alerts
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

    # Append COMBINED emotion to history and check alerts (CORRECTED)
    if combined_emotion:
        emotion_history.append(combined_emotion)
        if check_alerts(combined_emotion):  # Pass emotion directly to check_alerts
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
    # Append received emotion to history and check alerts (CORRECTED)
    if emotion:
        emotion_history.append(emotion)
        if check_alerts(emotion):  # Pass emotion directly to check_alerts
            emit('backend_status', {'message': "🚨 ALERT: SocketIO emotion fluctuation detected!"}, broadcast=True)
    
    emit('update_emotion', data, broadcast=True)

@app.route('/shutdown', methods=['POST'])
def shutdown():
    cap.release()
    return 'Server shutting down...'

if __name__ == '__main__':
    socketio.run(app, debug=True)
