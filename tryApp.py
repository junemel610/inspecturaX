import threading
from flask import Flask, Response
from flask_compress import Compress
import cv2
from roboflow import Roboflow
import time
import logging

# Initialize Flask app
app = Flask(__name__)
Compress(app)  # Enable compression for better performance

# Set up logging
logging.basicConfig(filename='wood_count.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

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

# Global variables for predictions and counts
predictions = []
lock = threading.Lock()  # Create a lock for thread synchronization
wood_count = 0  # Initialize wood count
detected_woods = {}  # To keep track of counted wood pieces and their defects
ROI = (0, 0, 640, 360)  # Default ROI, will be updated based on frame dimensions

def predict(frame):
    """ Predicts the objects in the frame using the Roboflow model. """
    cv2.imwrite("temp_frame.jpg", frame)  # Save the frame temporarily
    prediction = model.predict("temp_frame.jpg", confidence=80, overlap=30).json()
    print("Raw Predictions:", prediction)  # Print raw predictions
    return prediction

def prediction_thread():
    global predictions, wood_count, detected_woods, ROI
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Update ROI based on frame dimensions
        height, width = frame.shape[:2]
        ROI = (int(width * 0.15), int(height * 0.1), int(width * 0.85), int(height * 0.9))  # Adjust ROI dynamically
        x1, y1, x2, y2 = ROI
        
        # Crop the frame to the ROI
        frame_cropped = frame[y1:y2, x1:x2]

        preds = predict(frame_cropped)

        with lock:  # Acquire lock to update predictions safely
            predictions.clear()  # Clear previous predictions
            if 'predictions' in preds:
                for pred in preds['predictions']:
                    confidence = pred['confidence']
                    object_id = pred['class']  # Use class as ID for simplicity
                    defect_name = "Death_knot"  # Change this to get the actual defect name

                    print(f'Object ID: {object_id}, Confidence: {confidence}')  # Debugging line

                    # Check if the prediction has sufficient confidence
                    if confidence >= 0.80:  # Only confidence level checked
                        
                        # Create a unique identifier based on the bounding box coordinates
                        wood_piece_id = f"{object_id}_{int(pred['x'])}_{int(pred['y'])}"  # Unique ID based on position

                        # Count this wood piece if it hasn't been counted yet
                        if wood_piece_id not in detected_woods:
                            wood_count += 1  # Increment wood count
                            detected_woods[wood_piece_id] = {'defects': {defect_name: 1}}  # Initialize defects
                            logging.info(f'Wood counted: {wood_count} for wood piece {wood_piece_id} with defect {defect_name}')
                            print(f'Wood counted: {wood_count} for wood piece {wood_piece_id} with defect {defect_name}')  # Debugging line
                        else:
                            # Increment the defect count if the wood piece is already detected
                            if defect_name in detected_woods[wood_piece_id]['defects']:
                                detected_woods[wood_piece_id]['defects'][defect_name] += 1
                            else:
                                detected_woods[wood_piece_id]['defects'][defect_name] = 1

                        # Adjust coordinates based on the original frame
                        adjusted_x1 = int(pred['x'] * (x2 - x1) / 640) + x1
                        adjusted_y1 = int(pred['y'] * (y2 - y1) / 360) + y1
                        adjusted_width = int(pred.get('width', 0) * (x2 - x1) / 640)
                        adjusted_height = int(pred.get('height', 0) * (y2 - y1) / 360)
                        adjusted_x2 = adjusted_x1 + adjusted_width
                        adjusted_y2 = adjusted_y1 + adjusted_height
                        
                        # Append the prediction details
                        predictions.append({
                            'class': pred['class'],
                            'confidence': confidence,
                            'x1': adjusted_x1,
                            'y1': adjusted_y1,
                            'x2': adjusted_x2,
                            'y2': adjusted_y2
                        })

        # Wait for 3 seconds before processing the next frame
        time.sleep(3)

# Start the prediction thread
threading.Thread(target=prediction_thread, daemon=True).start()

def generate_frames():
    prev_time = time.time()
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Draw the ROI on the frame
        x1, y1, x2, y2 = ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw ROI in blue

        with lock:  # Acquire lock to read predictions safely
            # Display the total wood count on the frame
            cv2.putText(frame, f'Total Wood Count: {wood_count}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Only show the most recent wood detection
            if detected_woods:
                recent_wood_id = next(reversed(detected_woods))  # Get the most recent wood piece
                recent_data = detected_woods[recent_wood_id]
                total_defects = sum(recent_data['defects'].values())  # Total defects for the most recent wood piece

                # Display the wood number and total defects
                wood_no = wood_count  # Use the total wood count as the current wood number
                defect_info = ', '.join([f"{defect}: {count}" for defect, count in recent_data['defects'].items()])
                cv2.putText(frame, f'Wood No. {wood_no}: {defect_info}', (10, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw bounding boxes for all detected woods
            for pred in predictions:
                if pred['confidence'] >= 0.80:  # Draw only those with high confidence
                    cv2.rectangle(frame, (pred['x1'], pred['y1']), (pred['x2'], pred['y2']), (0, 0, 255), 2)  # Red color
                    cv2.putText(frame, f"{pred['class']} ({pred['confidence'] * 100:.1f}%)",
                                (pred['x1'], pred['y1'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red text

            # Display FPS in the bottom right corner
            fps = 1 / (time.time() - prev_time) if prev_time != time.time() else 0
            prev_time = time.time()
            cv2.putText(frame, f'FPS: {fps:.2f}', (frame.shape[1] - 100, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
    response.headers['bypass-tunnel-reminder'] = 'any_value'
    return response

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=8080)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()