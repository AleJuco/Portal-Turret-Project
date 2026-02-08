import cv2
import requests
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Obtained from webcam server on ESP-32 CAM
url = "http://192.168.113.222:81/stream"

# Download the face detection model if not present
# Using full-range model for better detection at distance
model_path = "blaze_face_short_range.tflite"
if not os.path.exists(model_path):
    print("Downloading face detection model (full range)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        model_path
    )

# Scale factor for detecting smaller/distant faces (1.5 = 50% larger)
SCALE_FACTOR = 1.0

# Display scale factor (2.0 = window twice as large)
DISPLAY_SCALE = 2.0

# Initialize MediaPipe Face Detection using Tasks API
base_options = python.BaseOptions(model_asset_path=model_path)
# Lower confidence threshold helps detect faces at distance
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.25)
face_detector = vision.FaceDetector.create_from_options(options)

print("--- Connecting to ESP32-CAM Stream ---")

try:
    # Setting stream=True is vital for MJPEG
    stream = requests.get(url, stream=True, timeout=10)
    if stream.status_code != 200:
        print(f"Error: Received status code {stream.status_code}")
        exit()
    print("Connection Successful. Press 'q' on the image window to stop.")
except Exception as e:
    print(f"Connection Failed: {e}")
    print("Check: Is the browser tab closed? Is the IP correct?")
    exit()

# Create a resizable window
cv2.namedWindow('Tracking Turret Feed', cv2.WINDOW_NORMAL)

bytes_data = bytes()
for chunk in stream.iter_content(chunk_size=1024):
    bytes_data += chunk
    a = bytes_data.find(b'\xff\xd8') # JPEG Start
    b = bytes_data.find(b'\xff\xd9') # JPEG End
    
    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]
        
        # Skip if jpg data is empty
        if len(jpg) == 0:
            continue
        
        # Convert raw bytes to OpenCV image
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if img is not None:
            # Scale up image for better detection at distance
            h_orig, w_orig = img.shape[:2]
            scaled_img = cv2.resize(img, (int(w_orig * SCALE_FACTOR), int(h_orig * SCALE_FACTOR)))
            
            # Convert BGR to RGB for MediaPipe
            rgb_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image from numpy array
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            
            # Detect faces using Tasks API
            detection_result = face_detector.detect(mp_image)
            
            # Draw face detections
            if detection_result.detections:
                for detection in detection_result.detections:
                    # Get bounding box and scale back to original image size
                    bbox = detection.bounding_box
                    x = int(bbox.origin_x / SCALE_FACTOR)
                    y = int(bbox.origin_y / SCALE_FACTOR)
                    w = int(bbox.width / SCALE_FACTOR)
                    h = int(bbox.height / SCALE_FACTOR)
                    
                    # Draw rectangle and label
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    confidence = int(detection.categories[0].score * 100)
                    cv2.putText(img, f'Face {confidence}%', (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Scale up for display
            display_img = cv2.resize(img, (int(w_orig * DISPLAY_SCALE), int(h_orig * DISPLAY_SCALE)))
            cv2.imshow('Tracking Turret Feed', display_img)
        else:
            print("Warning: Skipped a corrupted frame.")

        # If you don't include this, the window will never appear/refresh
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

