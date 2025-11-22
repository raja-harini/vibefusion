import numpy as np

def fuse_emotions(facial_emotion, speech_emotion, eeg_emotion):
    """
    Fuse emotion predictions from 3 modalities using simple majority vote.
    Returns the final predicted emotion.
    Handles numpy arrays and other non-string types by converting them to strings safely.
    """
    emotions = []
    for e in [facial_emotion, speech_emotion, eeg_emotion]:
        if e is not None:
            # Convert numpy arrays or others to string if needed
            if hasattr(e, 'tolist'):
                e = e.tolist() if isinstance(e, np.ndarray) else str(e)
                # If e is still a list, convert to string
                if isinstance(e, list):
                    e = str(e)
            else:
                e = str(e)
            emotions.append(e)

    if not emotions:
        return None

    final_emotion = max(set(emotions), key=emotions.count)
    return final_emotion
