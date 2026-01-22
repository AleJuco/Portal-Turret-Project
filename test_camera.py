import cv2

print("Scanning for available cameras...")
for idx in range(10):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ Camera {idx}: Working (Resolution: {frame.shape[1]}x{frame.shape[0]})")
        else:
            print(f"⚠ Camera {idx}: Opened but can't read frames")
        cap.release()
    else:
        print(f"✗ Camera {idx}: Not available")

print("\nDone! Use the index that shows '✓ Working' in your BodyTrack.py")
