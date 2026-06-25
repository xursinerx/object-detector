# Real-Time Object Detector

Webcam-based object detection using YOLOv8 and OpenCV. Identifies everyday objects and draws bounding boxes around them in real-time.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python
```

## Usage

**Static image detection:**
```bash
python detect_image.py
```

**Real-time webcam detection (coming soon):**
```bash
python detect_realtime.py
```

## How it works

- YOLOv8 nano model runs inference on each video frame
- OpenCV captures the webcam stream and displays annotated output
- Pre-trained on COCO dataset (80 object classes: people, vehicles, animals, household items, etc.)

## Controls

- Press `q` to quit the webcam feed
