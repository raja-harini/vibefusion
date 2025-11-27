import os
import smtplib
from email.message import EmailMessage
from collections import deque, Counter
from datetime import datetime
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

def alert_user_and_caregiver(emotion_counts, alert_type="fluctuation"):
    recipients = []
    if USER_EMAIL:
        recipients.append(USER_EMAIL)
    if CAREGIVER_EMAIL:
        recipients.append(CAREGIVER_EMAIL)

    if not recipients:
        print("Warning: No valid recipient emails set. Alert not sent.")
        return

    # Create user-friendly email in exact format
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if alert_type == "fluctuation":
        subject = '🚨 VibeFusion Emotional Alert 🚨'
        body_lines = [
            "Your emotional state shows significant fluctuations:",
        ]
        
        # Format emotion counts
        for emotion, count in emotion_counts.most_common():
            body_lines.append(f"• {emotion.capitalize()}: {count} times")
        
        body_lines.extend([
            "",
            "💡 Suggestions:",
            "• Your emotions are changing rapidly between multiple states",
            "",
            "Please take a moment to relax, breathe deeply, or talk to someone.",
            "",
            f"Time: {current_time}"
        ])
    
    elif alert_type == "sustained_negative":
        subject = '🚨 VibeFusion Negative Emotion Alert 🚨'
        body_lines = [
            "Sustained negative emotions detected:",
        ]
        
        negative_emotions = []
        for emotion, count in emotion_counts.items():
            if count > 0:
                negative_emotions.append(f"{emotion.capitalize()}: {count}")
        
        body_lines.extend([
            "• " + " | ".join(negative_emotions),
            "",
            "💡 Suggestions:",
            "• These emotions have persisted for some time",
            "• Consider relaxation techniques or speaking with someone",
            "",
            f"Time: {current_time}"
        ])
    
    body = "\n".join(body_lines)
    send_email_alert(subject, body, recipients)

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
    sustained_duration = 10   # Number of intervals for sustained negative emotion

    emotion_history.append(new_emotion)

    fluctuations = count_fluctuations(emotion_history)
    if fluctuations >= fluctuation_threshold:
        # Count recent emotions for user-friendly display
        recent_emotions = list(emotion_history)[-15:]  # Last 15 emotions
        emotion_counts = Counter(recent_emotions)
        alert_user_and_caregiver(emotion_counts, "fluctuation")
        # Clear history after alert to avoid repeated alerts
        emotion_history.clear()
        return True

    if sustained_negative(emotion_history, negative_emotions, sustained_duration):
        # Count negative emotions
        recent_emotions = list(emotion_history)[-15:]
        emotion_counts = Counter(e for e in recent_emotions if e in negative_emotions)
        alert_user_and_caregiver(emotion_counts, "sustained_negative")
        # Clear history to avoid repeated alerting
        emotion_history.clear()
        return True
    
    return False
