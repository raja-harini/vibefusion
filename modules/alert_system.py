import os
from collections import deque
from twilio.rest import Client
from dotenv import load_dotenv
load_dotenv() 
# Initialize emotion history buffer with max length 30
emotion_history = deque(maxlen=10)

# Define negative emotions to monitor
negative_emotions = ['sadness', 'anger', 'fear','disgust']

# Threshold parameters
fluctuation_threshold = 1  # Number of emotion changes within window triggering alert
sustained_duration = 5    # Number of intervals for sustained negative emotion

# Twilio credentials loaded from environment variables
account_sid = os.getenv('TWILIO_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_phone = os.getenv('TWILIO_PHONE')
user_phone = os.getenv('USER_PHONE')
caregiver_phone = os.getenv('CAREGIVER_PHONE')

client = Client(account_sid, auth_token)

def send_sms_alert(to_phone, message):
    try:
        msg = client.messages.create(
            body=message,
            from_=twilio_phone,
            to=to_phone
        )
        print(f"Sent SMS alert to {to_phone}: SID {msg.sid}")
    except Exception as e:
        print(f"Failed to send SMS to {to_phone}: {e}")

def send_alerts_to_user_and_caregiver(message):
    """
    Send SMS alerts to both user and caregiver phone numbers loaded from environment variables.
    """
    if user_phone:
        send_sms_alert(user_phone, message)
    if caregiver_phone:
        send_sms_alert(caregiver_phone, message)

def trigger_alert(message):
    """
    Trigger an alert: print locally and send SMS alerts to user and caregiver.
    """
    print(f"ALERT: {message}")  # Optional local logging
    send_alerts_to_user_and_caregiver(message)

def count_fluctuations(history):
    """
    Count how many times the emotion changes in the history deque.
    """
    changes = sum(1 for i in range(1, len(history)) if history[i] != history[i-1])
    return changes

def sustained_negative(history, neg_emotions, duration):
    """
    Check if the last 'duration' emotions in history are all negative emotions.
    """
    if len(history) < duration:
        return False
    recent = list(history)[-duration:]
    return all(em in neg_emotions for em in recent)

def check_alerts_with_emotion(emotion):
    """
    Add new emotion to history, check for alert conditions, trigger alerts if needed.
    """
    emotion_history.append(emotion)

    # Check conditions for alerting
    fluctuations = count_fluctuations(emotion_history)
    if fluctuations > fluctuation_threshold:
        trigger_alert(f"High emotional fluctuation detected: {list(emotion_history)}")
        emotion_history.clear()
        return True

    if sustained_negative(emotion_history, negative_emotions, sustained_duration):
        trigger_alert(f"Sustained negative emotion detected: {list(emotion_history)}")
        emotion_history.clear()
        return True

    return False
