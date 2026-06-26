# ABOUTME: Runs YOLOv8 object detection on a single image from URL.
# ABOUTME: Saves annotated result and prints detected objects with confidence scores.
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
results = model("https://i.etsystatic.com/26094858/r/il/610350/3096311722/il_fullxfull.3096311722_mb9m.jpg")
# results[0].show()
results[0].save(filename="result.jpg")

signal.signal(signal.SIGINT, signal.SIG_DFL)

for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    name = model.names[class_id]
    print(f"Detected: {name} ({confidence:.2f})")
