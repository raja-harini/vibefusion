import cv2
from deepface import DeepFace


def get_facial_emotion(frame):
    """
    Analyze the given webcam frame and return the detected facial emotion.
    Handles cases of multiple faces by extracting the first face emotion.
    """
    emotion = None
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        # Handle multiple faces or single face result:
        if isinstance(analysis, list):
            # Take first face's emotion
            emotion = analysis[0]['dominant_emotion']
        else:
            emotion = analysis['dominant_emotion']
    except Exception as e:
        print(f"Facial emotion detection error: {e}")
        emotion = None
    return emotion


def get_facial_emotion_from_webcam():
    # Open webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, None
    
    emotion = get_facial_emotion(frame)
    return frame, emotion


while True:
    # Get latest frame and emotion
    frame, emotion = get_facial_emotion_from_webcam()
    if frame is None:
        break

    if emotion:
        # Display emotion label on the frame
        cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Facial Emotion Recognition', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
