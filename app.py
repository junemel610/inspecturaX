from flask import Flask, Response
from flask_compress import Compress
import cv2
import numpy as np
import requests

# Initialize Flask app
app = Flask(__name__)
Compress(app) 

# Set up camera (use 0 for default camera)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # For Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Roboflow API setup
API_KEY = "RFaNdGHxTtn46bvxSFvM" 
PROJECT_NAME = "yolo-wood"
VERSION_NUMBER = 2

def predict(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'file': img_encoded.tobytes()}
    
    # Make a POST request to the Roboflow API
    response = requests.post(
        f'https://api.roboflow.com/{PROJECT_NAME}/version/{VERSION_NUMBER}/predict',
        files=files,
        headers={'Authorization': f'Bearer {API_KEY}'}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(data) 
        if 'predictions' in data:
            return data
        else:
            print("No predictions found in response:", data)
            return None
    else:
        print("Error:", response.status_code, response.text)
        return None

def generate_frames():
    frame_skip = 2  # Only process every 2nd frame to reduce load
    count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Skip frames to reduce processing load
        if count % frame_skip == 0:
            # Make a prediction
            predictions = predict(frame)

            # If predictions are available, draw them on the frame
            if predictions:
                for pred in predictions['predictions']:
                    x1, y1, x2, y2 = pred['x'], pred['y'], pred['x2'], pred['y2']
                    label = pred['class']
                    confidence = pred['confidence']

                    # Draw bounding box and label
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode frame in JPEG format with reduced quality
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame = buffer.tobytes()

            # Yield frame with multipart content type for video streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        count += 1

@app.route('/')
def index():
    return "Video feed endpoint accessed"

@app.route('/video_feed')
def video_feed():

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)