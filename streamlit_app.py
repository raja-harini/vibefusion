import streamlit as st
import cv2
import numpy as np
import base64
import time
import sys
import os
from datetime import datetime
from collections import deque
import threading

# Add current directory to system path to import modules (EXACTLY like vibefusion_main.py)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import YOUR exact modules
from modules.facial_emotion import get_facial_emotion
from modules.speech_emotion import get_speech_emotion
from modules.eeg_emotion import get_eeg_emotion
from fusion import fuse_emotions
from modules.alert_system import emotion_history, check_alerts

# Custom CSS (UNCHANGED - all your styling preserved)
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

# Page config (UNCHANGED)
st.set_page_config(page_title="VibeFusion", page_icon="😊", layout="wide")

# Global state (UNCHANGED structure)
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

# YOUR EXACT camera functions from app.py
@st.cache_resource
def init_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def capture_frame():
    if st.session_state.cap is None:
        st.session_state.cap = init_camera()
    with threading.Lock():
        ret, frame = st.session_state.cap.read()
    return frame if ret else None

def encode_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

# Main title (UNCHANGED)
st.markdown("<h1>VibeFusion Real-Time Emotion Recognition</h1>", unsafe_allow_html=True)

# Main layout (UNCHANGED)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Emotion Detection Controls")
    
    # Facial emotion button (INTEGRATED with YOUR modules)
    if st.button("🎭 Get Facial Emotion", key="facial_btn"):
        st.session_state.status_msg = "Capturing frame..."
        frame = capture_frame()
        if frame is not None:
            emotion = get_facial_emotion(frame)  # YOUR module
            st.session_state.status_msg = f"Facial: {emotion}"
            st.session_state.result_text = f"Facial Emotion: {emotion}"
            st.session_state.current_emotion = emotion
            st.session_state.face_image = encode_frame_to_base64(frame)
            st.session_state.facial_recognition_done = True
            
            # YOUR EXACT alert system integration
            if check_alerts(emotion):
                st.session_state.status_msg += " 🚨 ALERT TRIGGERED!"
            st.rerun()
    
    # Speech emotion button (INTEGRATED with YOUR modules)
    if st.button("🎤 Get Speech Emotion", key="speech_btn"):
        st.session_state.status_msg = "Recording speech..."
        emotion = get_speech_emotion()  # YOUR module
        st.session_state.status_msg = f"Speech: {emotion}"
        st.session_state.result_text = f"Speech Emotion: {emotion}"
        st.session_state.current_emotion = emotion
        
        # YOUR EXACT alert system integration
        if check_alerts(emotion):
            st.session_state.status_msg += " 🚨 ALERT TRIGGERED!"
        st.rerun()
    
    # EEG emotion button (INTEGRATED with YOUR modules)
    if st.button("🧠 Get EEG Emotion", key="eeg_btn"):
        st.session_state.status_msg = "Analyzing EEG..."
        emotion = get_eeg_emotion()  # YOUR module
        st.session_state.status_msg = f"EEG: {emotion}"
        st.session_state.result_text = f"EEG Emotion: {emotion}"
        st.session_state.current_emotion = emotion
        
        # YOUR EXACT alert system integration
        if check_alerts(emotion):
            st.session_state.status_msg += " 🚨 ALERT TRIGGERED!"
        st.rerun()
    
    # Combined emotion button (INTEGRATED with YOUR fuse_emotions)
    if st.button("🔗 Get Combined Emotion", key="combined_btn"):
        st.session_state.status_msg = "Analyzing all modalities..."
        frame = capture_frame()
        facial = get_facial_emotion(frame) if frame is not None else "neutral"
        speech = get_speech_emotion()
        eeg = get_eeg_emotion()
        combined = fuse_emotions(facial, speech, eeg)  # YOUR fusion module
        
        st.session_state.status_msg = f"Combined: {combined}"
        st.session_state.result_text = f"Facial: {facial}\nSpeech: {speech}\nEEG: {eeg}\nCombined: {combined}"
        st.session_state.current_emotion = combined
        if frame is not None:
            st.session_state.face_image = encode_frame_to_base64(frame)
            st.session_state.facial_recognition_done = True
        
        # YOUR EXACT alert system integration on COMBINED emotion
        if check_alerts(combined):
            st.session_state.status_msg += " 🚨 ALERT TRIGGERED!"
        st.rerun()

with col2:
    # Status (UNCHANGED)
    st.markdown(f'<div id="status">Status: {st.session_state.status_msg}</div>', unsafe_allow_html=True)
    
    # Result container (UNCHANGED)
    st.markdown(f'<div id="resultContainer">{st.session_state.result_text}</div>', unsafe_allow_html=True)
    
    # Face frame (UNCHANGED)
    if st.session_state.facial_recognition_done and st.session_state.face_image:
        st.markdown(f'<img id="faceFrame" src="data:image/jpeg;base64,{st.session_state.face_image}" />', unsafe_allow_html=True)
    else:
        st.markdown('<div style="height: 240px; display: flex; align-items: center; justify-content: center; color: #999; font-style: italic;">No facial recognition performed yet</div>', unsafe_allow_html=True)

# Real-time emotion display (UNCHANGED)
st.markdown('<h2 style="text-align:center; margin-top: 50px; color: #394f78; font-weight: 600;">Real-time Emotion Updates</h2>', unsafe_allow_html=True)
st.markdown(f'<div id="realTimeEmotion">{st.session_state.current_emotion}</div>', unsafe_allow_html=True)

# Emotion history and alerts (ENHANCED with YOUR alert_system)
st.subheader("📊 Recent Emotions & Alerts")
col3, col4 = st.columns(2)

with col3:
    recent_emotions = list(emotion_history)[-10:]  # YOUR shared emotion_history
    if recent_emotions:
        emotion_str = " → ".join(recent_emotions)
        st.caption(f"History (last 10): {emotion_str}")
    
    # YOUR alert system status display
    if len(emotion_history) >= 5:
        recent = list(emotion_history)[-5:]
        if len(set(recent)) >= 4:
            st.error("🚨 **ALERT: High emotional fluctuation detected!**")
            st.info("✅ Email alerts sent to user & caregiver")
        else:
            st.success("✅ Emotion stable")

with col4:
    if st.button("🗑️ Clear History", use_container_width=False):
        emotion_history.clear()  # YOUR shared emotion_history
        st.session_state.result_text = ""
        st.session_state.current_emotion = "History cleared"
        st.session_state.facial_recognition_done = False
        st.session_state.face_image = None
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.rerun()

# Live webcam preview section (UNCHANGED logic, YOUR camera handling)
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

# Live preview logic (INTEGRATED with YOUR modules)
if st.session_state.live_preview_active:
    frame_placeholder = st.empty()
    if st.session_state.cap is None:
        st.session_state.cap = init_camera()
    
    st.info("🔴 LIVE PREVIEW ACTIVE - Click STOP to end")
    
    preview_frames = st.empty()
    preview_frames.info("Preview running... (10 seconds)")
    
    for i in range(100):  # ~10 seconds
        if not st.session_state.live_preview_active:
            break
        
        frame = capture_frame()
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", width=400)
            
            try:
                emotion = get_facial_emotion(frame)  # YOUR module
                st.caption(f"Live emotion: {emotion}")
                
                # YOUR alert system in live preview
                if check_alerts(emotion):
                    st.error("🚨 Live alert triggered!")
            except:
                pass
        
        time.sleep(0.1)
    
    st.session_state.live_preview_active = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.rerun()

# Footer (UNCHANGED)
st.markdown("---")
st.markdown("*VibeFusion - Multimodal Emotion Recognition System*")
