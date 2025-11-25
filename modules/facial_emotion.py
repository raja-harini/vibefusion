import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained facial emotion model (run once at import)
model = load_model('models/facial_emotion_model.keras')

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def preprocess_face(frame):
    """Preprocess webcam frame for model input."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(gray, (48, 48))
    face_img = face_img.astype('float32') / 255.0  # normalize pixels
    face_img = np.expand_dims(face_img, axis=-1)   # add channel dimension
    face_img = np.expand_dims(face_img, axis=0)    # add batch dimension
    return face_img

def decode_prediction(preds):
    """Map class probabilities to emotion label."""
    return emotion_labels[np.argmax(preds)]

def predict_emotion(frame):
    """Predict emotion label from webcam frame using trained model."""
    input_data = preprocess_face(frame)
    preds = model.predict(input_data)
    emotion = decode_prediction(preds)
    return emotion

def get_facial_emotion(frame):
    """
    For backward compatibility: original DeepFace method.
    Can be removed or used as fallback.
    """
    emotion = None
    try:
        from deepface import DeepFace
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list):
            emotion = analysis[0]['dominant_emotion']
        else:
            emotion = analysis['dominant_emotion']
    except Exception as e:
        print(f"Facial emotion detection error: {e}")
        emotion = None
    return emotion

def get_facial_emotion_from_webcam():
    """Capture one frame from webcam and predict emotion using trained model."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, None

    # Use trained model instead of DeepFace
    emotion = predict_emotion(frame)
    return frame, emotion


if __name__ == "__main__":
    while True:
        frame, emotion = get_facial_emotion_from_webcam()
        if frame is None:
            break

        if emotion:
            cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Facial Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
