import cv2
import numpy as np
import imutils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Parameters for loading data and models
detection_model_path = 'haarcascade_frontalface_default.xml'  # Ensure this file is in your local directory
emotion_model_path = 'final_mini_XCEPTION_model_v2.keras'  # Adjust path to your local model file

# Load models
try:
    face_detection = cv2.CascadeClassifier(detection_model_path)
    if not face_detection.empty():
        print("Haar cascade loaded successfully")
    else:
        raise ValueError("Failed to load Haar cascade file")
except Exception as e:
    print(f"Error loading Haar cascade: {e}")
    exit()

try:
    emotion_classifier = load_model(emotion_model_path, compile=False)
    print("Emotion model loaded successfully")
except Exception as e:
    print(f"Error loading emotion model: {e}")
    exit()

# Define emotions
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Start video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)  # Use default webcam
if not camera.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Resize and convert frame
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()
    canvas = np.zeros((250, 300, 3), dtype="uint8")

    # Detect faces
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        # Sort faces by size and take the largest one
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces

        # Extract and preprocess ROI
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))  # Matches model input shape
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Predict emotions
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        # Draw probability bars
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # Draw face rectangle and label
        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    # Display output
    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()