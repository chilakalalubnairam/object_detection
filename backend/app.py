
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from detector import ObjectDetector
import os

app = Flask(__name__, static_folder="static")
detector = ObjectDetector()

@app.route("/detect", methods=["GET"])
def detect():
    image, labels = detector.detect()
    return jsonify({
        "image": image,
        "labels": labels
    })

@app.route("/image/<filename>")
def get_image(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(debug=True)