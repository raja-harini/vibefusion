from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from modules.facial_emotion import get_facial_emotion
from modules.speech_emotion import get_speech_emotion
from modules.eeg_emotion import get_eeg_emotion
from fusion import fuse_emotions
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize webcam for facial emotion once for reuse
cap = cv2.VideoCapture(0)

# Serve web page
@app.route('/')
def index():
    # Serve the frontend page
    return render_template('index.html')

# Endpoint to get current emotions as JSON
@app.route('/get-emotion', methods=['GET'])
def get_emotion():
    # Capture one frame from webcam for facial emotion
    ret, frame = cap.read()
    if not ret:
        return jsonify({'error': 'Camera error'}), 500

    # Get individual emotions
    facial_emotion = get_facial_emotion(frame)
    speech_emotion = get_speech_emotion()
    eeg_emotion = get_eeg_emotion()

    # Fuse emotions
    combined_emotion = fuse_emotions(facial_emotion, speech_emotion, eeg_emotion)

    # Return emotions as JSON
    return jsonify({
        'facial_emotion': facial_emotion,
        'speech_emotion': speech_emotion,
        'eeg_emotion': eeg_emotion,
        'combined_emotion': combined_emotion
    })

# SocketIO event to receive emotion and broadcast to clients in real-time
@socketio.on('send_emotion')
def handle_emotion(data):
    # Broadcast data to all connected clients
    emit('update_emotion', data, broadcast=True)

# Shutdown endpoint to release camera resource gracefully
@app.route('/shutdown', methods=['POST'])
def shutdown():
    cap.release()
    return 'Server shutting down...'

if __name__ == '__main__':
    socketio.run(app, debug=True)
