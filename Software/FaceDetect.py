import cv2
import numpy as np
import mediapipe as mp

url ="" # Url from ESP-32 CAM stream. We will decode the video and process it frame by frame.

mp_face_detection = mp.solutions.face_detection.FaceDetection() # Initializes the face detection model
mp_drawing = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0) # Creates a capture object

if not capture.isOpened():
    print("Error opening video stream") # In the future we can add voice lines from the game
    exit() 


while True:
    ret, frame = capture.read() # Reads the video stream frame by frame in the loop
    
    if not ret:
        print("Error reading video stream")
        break

    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Converts the frame to RGB for better processingb

    results = mp_face_detection.process(rgbFrame) # Processes the frame to detect faces

    if results.detections:
        for detection in results.detections:
            
            # --- NEW CODE START ---
            # Get the bounding box relative coordinates (numbers between 0 and 1)
            bbox = detection.location_data.relative_bounding_box
            
            # Calculate pixel coordinates
            h, w, c = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Calculate Center
            centerX = int(x + (width / 2))
            centerY = int(y + (height / 2))

            print(f"Face found at: X={centerX}, Y={centerY}")
            
            # Draw a circle at the center so you can see it
            cv2.circle(frame, (centerX, centerY), 5, (0, 0, 255), -1)
            # --- NEW CODE END ---

            mp_drawing.draw_detection(frame, detection) # Draws the detected faces on the frame

    cv2.imshow('Face Detection', frame) # Displays the processed frame

    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to exit the loop
        break

capture.release() # Releases the capture object
cv2.destroyAllWindows() # Closes all OpenCV windows