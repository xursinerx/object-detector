from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
results = model("https://i.etsystatic.com/26094858/r/il/610350/3096311722/il_fullxfull.3096311722_mb9m.jpg")
results[0].show()
results[0].save(filename="result.jpg")

for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    name = model.names[class_id]
    print(f"Detected: {name} ({confidence:.2f})")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()