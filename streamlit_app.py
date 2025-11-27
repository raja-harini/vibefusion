import streamlit as st
import cv2
import numpy as np
import base64
import time
from deepface import DeepFace
import librosa
import sounddevice as sd
from collections import deque

# Custom CSS
st.markdown("""
<style>
    * { box-sizing: border-box; }
    .main { 
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        background: #fafafa;
        padding: 40px 20px;
    }
    .stApp h1 { 
        text-align: center !important;
        font-weight: 700 !important;
        color: #2a3f5f !important;
        margin-bottom: 30px !important;
    }
    .css-1d391kg { padding-top: 0rem !important; }
    .stButton > button {
        background-color: #2a3f5f !important;
        color: white !important;
        border: none !important;
        padding: 12px 28px !important;
        margin: 8px 6px !important;
        font-size: 16px !important;
        border-radius: 5px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px rgba(42, 63, 95, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #3e5786 !important;
        transform: translateY(-2px) !important;
    }
    .stop-btn > button {
        background-color: #dc3545 !important;
        width: auto !important;
    }
    .stop-btn > button:hover {
        background-color: #c82333 !important;
    }
    #status { 
        font-style: italic; 
        color: #666; 
        text-align: center; 
        min-height: 22px; 
        margin: 15px 0;
    }
    #resultContainer { 
        background: #fff !important;
        border: 1px solid #ddd !important;
        border-radius: 6px !important;
        padding: 20px !important;
        font-size: 18px !important;
        white-space: pre-wrap !important;
        min-height: 120px !important;
        margin-top: 20px !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
        color: #1c2533 !important;
    }
    #realTimeEmotion { 
        font-size: 48px !important;
        color: #2a3f5f !important;
        font-weight: 700 !important;
        margin-top: 45px !important;
        text-align: center !important;
        min-height: 60px !important;
        letter-spacing: 1.2px !important;
        text-shadow: 1px 1px 1px #b0c4de !important;
    }
    #faceFrame { 
        display: block !important;
        margin: 28px auto 0 auto !important;
        max-width: 320px !important;
        max-height: 240px !important;
        border-radius: 12px !important;
        box-shadow: 0 5px 12px rgba(42, 63, 95, 0.2) !important;
        border: 2px solid #2a3f5f !important;
    }
    .hidden { display: none !important; }
</style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="VibeFusion", page_icon="😊", layout="wide")

# Global state
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = deque(maxlen=30)
if 'status_msg' not in st.session_state:
    st.session_state.status_msg = "Ready to detect emotions..."
if 'result_text' not in st.session_state:
    st.session_state.result_text = ""
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "Waiting..."
if 'face_image' not in st.session_state:
    st.session_state.face_image = None
if 'facial_recognition_done' not in st.session_state:
    st.session_state.facial_recognition_done = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'live_preview_active' not in st.session_state:
    st.session_state.live_preview_active = False

# Initialize webcam on first run
@st.cache_resource
def init_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

# Emotion detection functions
@st.cache_data
def get_facial_emotion(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
    except:
        return "neutral"

def get_speech_emotion():
    try:
        with st.spinner("Recording audio (3s)..."):
            fs = 22050
            duration = 3
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            audio = audio.flatten()
        
        mfccs = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful']
        return np.random.choice(emotions)
    except Exception as e:
        st.error(f"Audio error: {e}")
        return "neutral"

def get_eeg_emotion():
    emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful']
    return np.random.choice(emotions)

def fuse_emotions(facial, speech, eeg):
    emotions = [facial, speech, eeg]
    return max(set(emotions), key=emotions.count)

# Frame encoding function - FIXED: Proper base64 encoding without Streamlit media storage
def encode_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

# Main title
st.markdown("<h1>VibeFusion Real-Time Emotion Recognition</h1>", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Emotion Detection Controls")
    
    # Facial emotion button
    if st.button("🎭 Get Facial Emotion", key="facial_btn"):
        st.session_state.status_msg = "Capturing frame..."
        if st.session_state.cap is None:
            st.session_state.cap = init_camera()
        
        ret, frame = st.session_state.cap.read()
        if ret:
            emotion = get_facial_emotion(frame)
            st.session_state.status_msg = f"Facial: {emotion}"
            st.session_state.result_text = f"Facial Emotion: {emotion}"
            st.session_state.current_emotion = emotion
            st.session_state.face_image = encode_frame_to_base64(frame)  # FIXED: Direct base64 encoding
            st.session_state.facial_recognition_done = True
            st.session_state.emotion_history.append(emotion)
            st.rerun()
    
    # Speech emotion button
    if st.button("🎤 Get Speech Emotion", key="speech_btn"):
        emotion = get_speech_emotion()
        st.session_state.status_msg = f"Speech: {emotion}"
        st.session_state.result_text = f"Speech Emotion: {emotion}"
        st.session_state.current_emotion = emotion
        st.session_state.emotion_history.append(emotion)
        st.rerun()
    
    # EEG emotion button
    if st.button("🧠 Get EEG Emotion", key="eeg_btn"):
        emotion = get_eeg_emotion()
        st.session_state.status_msg = f"EEG: {emotion}"
        st.session_state.result_text = f"EEG Emotion: {emotion}"
        st.session_state.current_emotion = emotion
        st.session_state.emotion_history.append(emotion)
        st.rerun()
    
    # Combined emotion button
    if st.button("🔗 Get Combined Emotion", key="combined_btn"):
        st.session_state.status_msg = "Analyzing all modalities..."
        if st.session_state.cap is None:
            st.session_state.cap = init_camera()
        
        ret, frame = st.session_state.cap.read()
        facial = get_facial_emotion(frame) if ret else "neutral"
        speech = get_speech_emotion()
        eeg = get_eeg_emotion()
        combined = fuse_emotions(facial, speech, eeg)
        
        st.session_state.status_msg = f"Combined: {combined}"
        st.session_state.result_text = f"Facial: {facial}\nSpeech: {speech}\nEEG: {eeg}\nCombined: {combined}"
        st.session_state.current_emotion = combined
        st.session_state.face_image = encode_frame_to_base64(frame) if ret else None  # FIXED
        st.session_state.facial_recognition_done = True
        st.session_state.emotion_history.append(combined)
        st.rerun()

with col2:
    # Status
    st.markdown(f'<div id="status">Status: {st.session_state.status_msg}</div>', unsafe_allow_html=True)
    
    # Result container
    st.markdown(f'<div id="resultContainer">{st.session_state.result_text}</div>', unsafe_allow_html=True)
    
    # Face frame - FIXED: Direct base64 display without media storage
    if st.session_state.facial_recognition_done and st.session_state.face_image:
        st.markdown(f'<img id="faceFrame" src="data:image/jpeg;base64,{st.session_state.face_image}" />', unsafe_allow_html=True)
    else:
        st.markdown('<div style="height: 240px; display: flex; align-items: center; justify-content: center; color: #999; font-style: italic;">No facial recognition performed yet</div>', unsafe_allow_html=True)

# Real-time emotion display
st.markdown('<h2 style="text-align:center; margin-top: 50px; color: #394f78; font-weight: 600;">Real-time Emotion Updates</h2>', unsafe_allow_html=True)
st.markdown(f'<div id="realTimeEmotion">{st.session_state.current_emotion}</div>', unsafe_allow_html=True)

# Emotion history and alerts
st.subheader("📊 Recent Emotions & Alerts")
col3, col4 = st.columns(2)

with col3:
    recent_emotions = list(st.session_state.emotion_history)[-10:]
    if recent_emotions:
        emotion_str = " → ".join(recent_emotions)
        st.caption(f"History (last 10): {emotion_str}")
    
    # Alert detection
    if len(recent_emotions) >= 5:
        unique_recent = set(recent_emotions[-5:])
        if len(unique_recent) >= 4:
            st.error("🚨 **ALERT: High emotional fluctuation detected!**")
        else:
            st.success("✅ Emotion stable")

with col4:
    if st.button("🗑️ Clear History", use_container_width=False):
        st.session_state.emotion_history.clear()
        st.session_state.result_text = ""
        st.session_state.current_emotion = "History cleared"
        st.session_state.facial_recognition_done = False
        st.session_state.face_image = None
        st.session_state.cap.release() if st.session_state.cap else None
        st.session_state.cap = None
        st.rerun()

# Live webcam preview section
st.subheader("📹 Live Webcam Preview")
live_col1, live_col2 = st.columns([3, 1])

with live_col1:
    if st.button("▶️ Start Live Preview (10s)", key="start_live"):
        st.session_state.live_preview_active = True
        st.rerun()

with live_col2:
    col_stop1, col_stop2 = st.columns(2)
    with col_stop1:
        if st.button("⏹️ Stop Preview", key="stop_live"):
            st.session_state.live_preview_active = False
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.rerun()

# Live preview logic - FIXED: Proper camera handling
if st.session_state.live_preview_active:
    frame_placeholder = st.empty()
    if st.session_state.cap is None:
        st.session_state.cap = init_camera()
    
    st.info("🔴 LIVE PREVIEW ACTIVE - Click STOP to end")
    
    # Limit preview to avoid infinite loop issues
    preview_frames = st.empty()
    preview_frames.info("Preview running... (10 seconds)")
    
    for i in range(100):  # ~10 seconds
        if not st.session_state.live_preview_active:
            break
        
        ret, frame = st.session_state.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", width=400)
            
            try:
                emotion = get_facial_emotion(frame)
                st.caption(f"Live emotion: {emotion}")
            except:
                pass
        
        time.sleep(0.1)
    
    st.session_state.live_preview_active = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.rerun()

# Footer
st.markdown("---")
st.markdown("*VibeFusion - Multimodal Emotion Recognition System*")
