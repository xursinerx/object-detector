# Real-Time Object Detector

Object detection using YOLOv8 and OpenCV. Identifies everyday objects and draws bounding boxes around them.

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

**Video detection:**
```bash
python detect_video.py
```
Processes a video file frame by frame and writes annotated output to `output.mp4`.

**Webcam (when available):**
In `detect_video.py`, change `VideoCapture("path")` to `VideoCapture(0)` and swap `out.write` for `cv2.imshow`. See comments in the file.

## How it works

- YOLOv8 nano model runs inference on each frame
- OpenCV handles video capture and output
- Pre-trained on COCO dataset (80 object classes: people, vehicles, animals, household items, etc.)
