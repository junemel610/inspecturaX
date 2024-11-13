import cv2
from roboflow import Roboflow
import time

# Set up camera (use 0 for default camera)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # For Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Roboflow API setup
API_KEY = "RFaNdGHxTtn46bvxSFvM"  # Replace with your actual API key
WORKSPACE = "yolo-wood"
MODEL_ENDPOINT = "project-design-ekhku"
VERSION = 2

# Initialize Roboflow model
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(MODEL_ENDPOINT)
model = project.version(VERSION).model

# Function to predict objects in the frame
def predict(frame):
    """ Predicts the objects in the frame using the Roboflow model. """
    _, img_encoded = cv2.imencode('.jpg', frame)
    temp_image_path = "temp_frame.jpg"
    cv2.imwrite(temp_image_path, frame)  # Save the frame temporarily

    # Make a prediction using the Roboflow model with increased confidence threshold
    prediction = model.predict(temp_image_path, confidence=60, overlap=30).json()  # Adjust to 60
    return prediction

# Main processing loop
prev_time = time.time()
while True:
    success, frame = cap.read()
    if not success:
        break

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (320, 240))

    # Get predictions
    preds = predict(frame_resized)
    if 'predictions' in preds:
        for pred in preds['predictions']:
            x1, y1 = pred['x'], pred['y']
            width, height = pred.get('width', 0), pred.get('height', 0)
            x2 = x1 + width  # Calculate x2
            y2 = y1 + height  # Calculate y2

            label = pred['class']
            confidence = pred['confidence']

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()