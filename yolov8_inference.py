import cv2
import torch

model = torch.hub.load("ultralytics/yolov8", "custom", path="best.pt")

def detect_objects(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        yield detections, frame
    cap.release()
