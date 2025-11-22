import cv2
from deepface import DeepFace

def get_facial_emotion():
    # Open webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, None
    
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = analysis['dominant_emotion']
    except:
        emotion = None

    return frame, emotion

while True:
    # Get latest frame and emotion
    frame, emotion = get_facial_emotion()
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
