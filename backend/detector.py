import cv2
import pyttsx3
import time
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        self.model = YOLO("yolov8s.pt")
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

        self.last_spoken = ""
        self.last_time = 0

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def detect(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None, []

        results = self.model(frame, conf=0.3)

        labels = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                labels.append(label)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

        if labels:
            current_time = time.time()
            if labels[0] != self.last_spoken or current_time - self.last_time > 3:
                self.speak(labels[0])
                self.last_spoken = labels[0]
                self.last_time = current_time

        cv2.imwrite("static/detected.jpg", frame)

        return "detected.jpg", labels