import threading
from flask import Flask, Response, jsonify
from flask_compress import Compress
import cv2
from roboflow import Roboflow
import time

# Initialize Flask app
app = Flask(__name__)
Compress(app)  # Enable compression for better performance

# Set up camera
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

# Global variable for predictions
predictions = []

def predict(frame):
    """ Predicts the objects in the frame using the Roboflow model. """
    _, img_encoded = cv2.imencode('.jpg', frame)
    temp_image_path = "temp_frame.jpg"
    cv2.imwrite(temp_image_path, frame)  # Save the frame temporarily

    # Make a prediction using the Roboflow model
    prediction = model.predict(temp_image_path, confidence=60, overlap=30).json()
    return prediction

def prediction_thread():
    global predictions
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (640, 320))

        # Measure prediction time
        start_time = time.time()
        preds = predict(frame_resized)
        print(f"Prediction time: {time.time() - start_time} seconds")

        if 'predictions' in preds:
            predictions = preds['predictions']

# Start the prediction thread
threading.Thread(target=prediction_thread, daemon=True).start()

def generate_frames():
    prev_time = time.time()
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Draw predictions on the frame
        for pred in predictions:
            x1, y1 = pred['x'], pred['y']
            width, height = pred.get('width', 0), pred.get('height', 0)
            x2 = x1 + width
            y2 = y1 + height

            label = pred['class']
            confidence = pred['confidence']

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.after_request
def after_request(response):
    # Add custom headers to the response
    response.headers['bypass-tunnel-reminder'] = 'any_value'
    return response

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=8080)