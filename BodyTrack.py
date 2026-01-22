import serial
import time
import numpy as np 
import cv2

ESP32Data = serial.Serial('COM1', 115200, timeout=1)

def send_Coords(x,y, w, h):

    coordinates = f"{x},{y}\n"
    ESP32Data.write(coordinates.encode())
    print(f"X: {x}, Y: {y}")

capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    isTrue, frame = capture.read()

    if not isTrue or frame is None:
        print("Error: Could not read frame from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
        send_Coords(x, y, w, h)

    cv2.imshow('Video', frame)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv2.destroyAllWindows()