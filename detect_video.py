# ABOUTME: Runs YOLOv8 object detection on each frame of a video file.
# ABOUTME: Writes annotated output with bounding boxes to output.mp4.
import cv2
from ultralytics import YOLO
# import signal
# signal needed for imshow

model = YOLO("yolov8n.pt")

# When webcam access, replace the filepath below with just 0
cap = cv2.VideoCapture("/home/analyst/object-detector/test_video.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

success, frame = cap.read()
while success:
    results = model(frame)
    annotated = results[0].plot()
    out.write(annotated)    # webcam: replace with cv2.imshow("Webcam", annotated)
    success, frame = cap.read()

cap.release()
out.release()