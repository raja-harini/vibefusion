import cv2
import os
import numpy as np

def preprocess_facial(video_path, output_dir, face_cascade_path='haarcascade_frontalface_default.xml'):
    os.makedirs(output_dir, exist_ok=True)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)
    cap = cv2.VideoCapture(video_path)
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face_img = gray_frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            np.save(os.path.join(output_dir, f"{count}.npy"), face_img)
            count += 1

    cap.release()
    print(f"Processed and saved {count} face frames from {video_path}")

def preprocess_all_videos(input_dir, output_root):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                # Create corresponding output directory preserving relative path
                relative_path = os.path.relpath(root, input_dir)
                output_dir = os.path.join(output_root, relative_path, os.path.splitext(file)[0])
                print(f"Processing {video_path} -> {output_dir}")
                preprocess_facial(video_path, output_dir)

if __name__ == '__main__':
    meld_train_dir = 'data/meld_train_subset'
    processed_output_root = 'data/processed/facial/train'
    preprocess_all_videos(meld_train_dir, processed_output_root)

    meld_test_dir = 'data/meld_test_subset'
    processed_test_output = 'data/processed/facial/test'
    preprocess_all_videos(meld_test_dir, processed_test_output)

    meld_train_dir = 'data/meld_train_val_subset'
    processed_output_root = 'data/processed/facial/val'
    preprocess_all_videos(meld_train_dir, processed_output_root)