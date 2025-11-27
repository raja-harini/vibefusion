import os
import smtplib
from email.message import EmailMessage
from collections import deque
from dotenv import load_dotenv

# Load .env file variables into environment
load_dotenv(dotenv_path='dont1.env')

# Load email credentials and recipients from environment variables
EMAIL_ADDRESS = os.getenv('ALERT_EMAIL_ADDRESS')      # Your email address (sender)
EMAIL_PASSWORD = os.getenv('ALERT_EMAIL_PASSWORD')    # App password or actual password
USER_EMAIL = os.getenv('USER_EMAIL')                  # User's email
CAREGIVER_EMAIL = os.getenv('CAREGIVER_EMAIL')        # Caregiver's email

# Keep recent emotions for fluctuation monitoring (e.g., last 30 predictions)
emotion_history = deque(maxlen=30)


def send_email_alert(subject, body, to_emails):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = ", ".join(to_emails) if isinstance(to_emails, list) else to_emails
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"Alert email sent to: {msg['To']}")
    except Exception as e:
        print(f"Error sending alert email: {e}")


def alert_user_and_caregiver(message):
    recipients = []
    if USER_EMAIL:
        recipients.append(USER_EMAIL)
    if CAREGIVER_EMAIL:
        recipients.append(CAREGIVER_EMAIL)

    if not recipients:
        print("Warning: No valid recipient emails set. Alert not sent.")
        return

    send_email_alert('VibeFusion Emotional Alert', message, recipients)


def count_fluctuations(history):
    """Count number of emotion changes in the history."""
    if len(history) < 2:
        return 0
    return sum(1 for i in range(1, len(history)) if history[i] != history[i-1])


def sustained_negative(history, negative_emotions, duration=10):
    """Check if negative emotions persist for given duration."""
    if len(history) < duration:
        return False
    recent = list(history)[-duration:]
    return all(em in negative_emotions for em in recent)


def check_alerts(new_emotion):
    """Add new emotion to history and check if alert condition meets."""
    negative_emotions = ['sadness', 'anger', 'fear']  # Customize as per your labels
    fluctuation_threshold = 3  # Number of changes to trigger alert
    sustained_duration = 10    # Number of intervals for sustained negative emotion

    emotion_history.append(new_emotion)

    fluctuations = count_fluctuations(emotion_history)
    if fluctuations >= fluctuation_threshold:
        alert_user_and_caregiver(f"High emotional fluctuation detected: {list(emotion_history)}")
        # Clear history after alert to avoid repeated alerts
        emotion_history.clear()
        return

    if sustained_negative(emotion_history, negative_emotions, sustained_duration):
        alert_user_and_caregiver(f"Sustained negative emotion detected: {list(emotion_history)}")
        # Clear history to avoid repeated alerting
        emotion_history.clear()
