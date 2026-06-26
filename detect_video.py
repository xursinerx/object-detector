# ABOUTME: Runs YOLOv8 object detection and tracking on each frame of a video file.
# ABOUTME: Features: confidence/class filtering, FPS overlay, persistent object IDs.
import cv2
from ultralytics import YOLO
import argparse
import time
import signal
# signal needed for imshow

model = YOLO("yolov8n.pt")

parser = argparse.ArgumentParser()
parser.add_argument("--conf", type=float, default=0.6, help="Minimum confidence threshold")
parser.add_argument("--classes", type=int, nargs="+", default=[0], help="Class IDs to detect")
parser.add_argument("--list-classes", action="store_true", help="Print all class IDs and exit")
args = parser.parse_args()

if args.list_classes:
    for id, name in model.names.items():
        print(f"{id}: {name}")
    exit()

# When webcam access, replace the filepath below with just 0
cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

line_y = height // 2
crossed_ids = set()
prev_centers = {}

success, frame = cap.read()
while success:
    start = time.time()
    results = model.track(frame, conf=args.conf, classes=args.classes, persist=True)
    for box in results[0].boxes:
        if box.id is None:
            continue
        track_id = int(box.id[0])
        x1, y1, x2, y2 = box.xyxy[0]
        center_y = int((y1 + y2) / 2)

        if track_id in prev_centers:
            if prev_centers[track_id] < line_y <= center_y:
                crossed_ids.add(track_id)

        prev_centers[track_id] = center_y
    end = time.time()

    fps_text = f"FPS: {1 / (end - start):.1f}"
    annotated = results[0].plot()

    cv2.line(annotated, (0, line_y), (width, line_y), (0, 0, 255), 2)
    cv2.putText(annotated, f"Count: {len(crossed_ids)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # out.write(annotated)    
    # webcam: replace with cv2.imshow("Webcam", annotated)
    cv2.imshow("Webcam", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    success, frame = cap.read()

cap.release()
#out.release()