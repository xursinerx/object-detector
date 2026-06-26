# ABOUTME: Serves real-time object detection via browser using Flask MJPEG streaming.
# ABOUTME: Features: tracking, zone counting, FPS overlay, accessible from any device on the network.
from flask import Flask, Response
import cv2
from ultralytics import YOLO
import argparse
import time

app = Flask(__name__)

model = YOLO("yolov8n.pt")

parser = argparse.ArgumentParser()
parser.add_argument("--conf", type=float, default=0.6)
parser.add_argument("--classes", type=int, nargs="+", default=[0])
args = parser.parse_args()

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

line_x = width // 2
entered_ids = set()
left_ids = set()
prev_centers = {}


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        start = time.time()
        results = model.track(frame, conf=args.conf, classes=args.classes, persist=True)
        for box in results[0].boxes:
            if box.id is None:
                continue
            track_id = int(box.id[0])
            x1, y1, x2, y2 = box.xyxy[0]
            center_x = int((x1 + x2) / 2)

            if track_id in prev_centers:
                if prev_centers[track_id] < line_x <= center_x:
                    entered_ids.add(track_id)
                elif prev_centers[track_id] > line_x >= center_x:
                    left_ids.add(track_id)

            prev_centers[track_id] = center_x
        end = time.time()

        fps_text = f"FPS: {1 / (end - start):.1f}"
        annotated = results[0].plot()

        cv2.line(annotated, (line_x, 0), (line_x, height), (0, 0, 255), 2)
        occupancy = len(entered_ids) - len(left_ids)
        cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"In: {len(entered_ids)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"Out: {len(left_ids)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated, f"Inside: {occupancy}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


        ret, buffer = cv2.imencode('.jpg', annotated)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<html><body><img src="/video"></body></html>'

app.run(host='0.0.0.0', port=5000)