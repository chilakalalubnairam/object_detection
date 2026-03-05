from flask import Flask, Response, render_template, jsonify
import cv2
from ultralytics import YOLO
import pyttsx3
import time
import os

app = Flask(__name__)

# -------------------- MODEL --------------------
model = YOLO("yolov8s.pt")   # Better accuracy

# -------------------- CAMERA --------------------
camera = None
camera_running = False

# -------------------- SPEECH ENGINE --------------------
engine = pyttsx3.init(driverName='espeak')
engine.setProperty('rate', 130)   # Slow and clear speech
engine.setProperty('volume', 1.0)

last_spoken = ""
last_spoken_time = 0


# -------------------- ALERT SOUND --------------------
def play_alert():
    # Linux terminal beep sound
    os.system('echo -e "\a"')


# -------------------- SPEAK FUNCTION --------------------
def speak(text):
    global last_spoken, last_spoken_time, camera_running

    if not camera_running:
        return

    current_time = time.time()

    # Avoid repeating same speech quickly
    if text != last_spoken and current_time - last_spoken_time > 3:
        engine.stop()
        engine.say(text)
        engine.runAndWait()
        last_spoken = text
        last_spoken_time = current_time


# -------------------- VIDEO GENERATOR --------------------
def generate_frames():
    global camera_running, camera

    while True:

        if not camera_running:
            time.sleep(0.1)
            continue

        if camera is None:
            time.sleep(0.1)
            continue

        success, frame = camera.read()
        if not success:
            continue

        results = model(frame)
        names = results[0].names
        boxes = results[0].boxes

        detected = []

        for box in boxes:
            confidence = float(box.conf[0])

            if confidence > 0.6:
                cls = int(box.cls[0])
                label = names[cls]

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                box_width = (x2 - x1)

                if box_width > 0:
                    distance = round(500 / box_width, 2)

                    # --------- OBSTACLE WARNING ----------
                    if distance < 0.8:
                        distance_text = "Very Close"
                        play_alert()
                        speech_text = f"Warning! {label} very close"
                    elif distance < 2:
                        distance_text = "Near"
                        speech_text = f"{label} ahead, near"
                    else:
                        distance_text = "Far"
                        speech_text = f"{label} ahead, far"

                    detected.append(speech_text)

                    # --------- DRAW VISUAL TEXT ----------
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cv2.putText(
                        frame,
                        f"{label} - {distance_text}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

        if detected:
            text = " . ".join(set(detected))
            speak(text)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -------------------- ROUTES --------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start')
def start_camera():
    global camera, camera_running

    if not camera_running:
        camera = cv2.VideoCapture(0, cv2.CAP_V4L2)

        if not camera.isOpened():
            return jsonify({"status": "camera_error"})

        camera_running = True

    return jsonify({"status": "started"})


@app.route('/stop')
def stop_camera():
    global camera_running, camera

    camera_running = False

    if camera:
        camera.release()

    engine.stop()  # Stop speech immediately

    return jsonify({"status": "stopped"})


# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)