# Real-Time Object Detector

Object detection and tracking using YOLOv8 and OpenCV. Identifies everyday objects, assigns persistent IDs, counts room entry/exit, and streams to a browser.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python flask mediapipe
```

## Usage

**Static image detection:**
```bash
python detect_image.py
```
Runs detection on a sample image URL and saves the annotated result to `result.jpg`.

**Video/webcam detection (OpenCV window):**
```bash
python detect_video.py
python detect_video.py --conf 0.5 --classes 0 2
python detect_video.py --list-classes
```
Processes video with object tracking. Press `q` to quit.

**Hand gesture detection (static image):**
```bash
python detect_gestures.py
```
Detects hands in `hand.jpg` and saves annotated result to `hand_result.jpg`.

**Combined person + gesture detection (webcam):**
```bash
python detect_combined.py
```
Runs YOLO person detection and MediaPipe hand gesture recognition simultaneously. Press `q` to quit.

**Web interface (browser stream):**
```bash
python web_app.py
python web_app.py --conf 0.5 --classes 0 2
```
Open `http://localhost:5000` in a browser. Accessible from other devices on the same network via your machine's IP.

**Options:**
- `--conf` - Minimum confidence threshold (default: 0.6)
- `--classes` - Class IDs to detect (default: 0/person, use --list-classes to see all)
- `--list-classes` - Print all 80 COCO class IDs and exit

## Features

- YOLOv8 nano model with persistent object tracking
- Configurable confidence threshold and class filtering
- FPS overlay
- Zone counting with entry/exit tracking and occupancy display
- Browser-based MJPEG streaming via Flask
- Pre-trained on COCO dataset (80 object classes)
- Hand gesture recognition via MediaPipe (open hand, fist, point, peace, gun)
- Combined person detection + gesture recognition
