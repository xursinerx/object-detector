# ABOUTME: Serves real-time object detection via browser using Flask MJPEG streaming.
# ABOUTME: Features: tracking, FPS overlay, accessible from any device on the network.
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
parser.add_argument("--sources", nargs='+', default=["0"])
args = parser.parse_args()

sources = [int(s) if s.isdigit() else s for s in args.sources]
caps = [cv2.VideoCapture(src) for src in sources]

def generate_frames(cap_obj):
    while True:
        success, frame = cap_obj.read()
        if not success:
            break
        start = time.time()
        results = model(frame, conf=args.conf, classes=args.classes)

        end = time.time()

        fps_text = f"FPS: {1 / (end - start):.1f}"
        annotated = results[0].plot()

        occupancy = len(results[0].boxes)
        cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"Inside: {occupancy}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


        ret, buffer = cv2.imencode('.jpg', annotated)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video/<int:idx>')
def video(idx):
    return Response(generate_frames(caps[idx]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    cards = ''
    for i in range(len(caps)):
        cards += f'''
        <div class="feed">
            <img src="/video/{i}">
            <p>Camera {i}</p>
        </div>'''

    return f'''<html>
<head>
    <title>Object Detector</title>
    <style>
        body {{
            background: #1a1a1a;
            color: white;
            font-family: sans-serif;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 20px;
        }}
        .feed {{
            text-align: center;
        }}
        .feed img {{
            width: 100%;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <h1>Object Detector</h1>
    <div class="grid">{cards}</div>
</body>
</html>'''

app.run(host='0.0.0.0', port=5000)
