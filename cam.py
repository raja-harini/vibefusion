# import cv2

# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open webcam")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break
#     cv2.imshow("Test Webcam", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

from modules.alert_system import alert_user_and_caregiver

test_message = "Test alert from VibeFusion system."
alert_user_and_caregiver(test_message)

# import os
# from dotenv import load_dotenv

# # Explicitly load your custom env file
# load_dotenv(dotenv_path='dont1.env')

# print("ALERT_EMAIL_ADDRESS:", os.getenv('ALERT_EMAIL_ADDRESS'))
# print("ALERT_EMAIL_PASSWORD:", os.getenv('ALERT_EMAIL_PASSWORD'))
# print("USER_EMAIL:", os.getenv('USER_EMAIL'))
# print("CAREGIVER_EMAIL:", os.getenv('CAREGIVER_EMAIL'))
