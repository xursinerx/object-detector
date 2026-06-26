# Real-Time Object Detector

Object detection and tracking using YOLOv8 and OpenCV. Identifies everyday objects, assigns persistent IDs, and draws bounding boxes.

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
Runs detection on a sample image URL and saves the annotated result to `result.jpg`.

**Video detection with tracking:**
```bash
python detect_video.py
python detect_video.py --conf 0.5 --classes 0 2
python detect_video.py --list-classes
```
Processes a video file frame by frame with object tracking and writes annotated output to `output.mp4`.

**Options:**
- `--conf` - Minimum confidence threshold (default: 0.6)
- `--classes` - Class IDs to detect (default: 0/person, use --list-classes to see all)
- `--list-classes` - Print all 80 COCO class IDs and exit

**Webcam (when available):**
In `detect_video.py`, change `VideoCapture("path")` to `VideoCapture(0)` and swap `out.write` for `cv2.imshow`. See comments in the file.

## Features

- YOLOv8 nano model with persistent object tracking
- Configurable confidence threshold and class filtering
- FPS overlay on output video
- Pre-trained on COCO dataset (80 object classes: people, vehicles, animals, household items, etc.)
