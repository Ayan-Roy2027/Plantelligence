import base64
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

WEIGHTS = os.environ.get("WEIGHTS", "best.pt")
CONF = 0.25  # minimum confidence threshold

app = Flask(__name__, static_folder=".")
CORS(app, resources={r"/*": {"origins": "*"}})
model = YOLO(WEIGHTS)
model.to('cpu')

# A class counts as "diseased" if its name contains one of these words.
DISEASE_WORDS = ["blight", "spot", "rust", "scab", "mildew", "mold", "mosaic",
                 "virus", "rot", "mite", "bacterial"]

ADVICE = {
    "blight": ["Remove and bin affected leaves (do not compost).",
               "Water at the base, not over the leaves.",
               "Improve airflow between plants.",
               "Ask a local agri shop about a suitable fungicide if it spreads."],
    "spot": ["Remove spotted leaves.", "Avoid wetting the leaves when watering.",
             "Clean tools after use.", "Use a suitable fungicide if spots keep spreading."],
    "mold": ["Reduce humidity and increase ventilation.", "Remove affected leaves.",
             "Avoid overcrowding the plants."],
    "mildew": ["Remove affected leaves.", "Improve sunlight and airflow.",
               "Avoid watering in the evening."],
    "rust": ["Remove infected leaves.", "Keep foliage dry.",
             "Use a rust-suitable fungicide if it spreads."],
    "scab": ["Remove fallen and infected leaves.", "Prune for better airflow."],
    "rot": ["Remove infected parts.", "Do not overwater; make sure soil drains well."],
    "virus": ["Isolate or remove the plant; viruses have no cure.",
              "Control insects such as whiteflies and aphids.",
              "Wash hands and tools after touching it."],
    "mosaic": ["Isolate or remove the plant; viruses have no cure.",
               "Control insects such as aphids.", "Wash hands and tools after touching it."],
    "mite": ["Spray leaves with water to wash mites off.",
             "Use neem oil or a miticide if it continues."],
    "bacterial": ["Remove infected leaves.", "Avoid overhead watering.",
                  "Do not work among wet plants."],
}
DEFAULT_ADVICE = ["Remove the affected leaves.", "Isolate the plant from others.",
                  "Show the plant to a local agriculture expert."]


def is_diseased(name: str) -> bool:
    n = name.lower()
    return any(w in n for w in DISEASE_WORDS)


def advice_for(name: str):
    n = name.lower()
    for key, tips in ADVICE.items():
        if key in n:
            return tips
    return DEFAULT_ADVICE


def to_data_url(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@app.get("/")
def index():
    return send_from_directory(".", "index.html")


@app.get("/health")
def health():
    return jsonify(ok=True, weights=WEIGHTS)


@app.post("/predict")
def predict():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="Send the photo in a form field named 'image'."), 400
    try:
        img = Image.open(f.stream).convert("RGB")
    except Exception:
        return jsonify(error="That file could not be read as an image."), 400

    result = model.predict(np.array(img), conf=CONF, verbose=False)[0]
    names = result.names

    detections = []
    for b in result.boxes:
        name = names[int(b.cls)]
        detections.append({"label": name, "confidence": round(float(b.conf), 4),
                           "box": [round(float(v), 1) for v in b.xyxy[0].tolist()],
                           "diseased": is_diseased(name)})
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    diseased = [d for d in detections if d["diseased"]]

    # Strictly binary output: Bad leaf vs Good leaf
    if diseased:
        status, top = "unhealthy", diseased[0]
        tips = advice_for(top["label"])
    elif detections:
        status, top = "healthy", detections[0]
        tips = ["No action needed.", "Keep checking leaves once a week."]
    else:
        # Fallback when no bounding box is detected: treat whole leaf as healthy
        status = "healthy"
        top = {"label": "Healthy Leaf", "confidence": 0.95}
        tips = ["No disease spotted. Plant appears healthy.", "Keep checking leaves once a week."]

    annotated = to_data_url(result.plot())

    return jsonify(status=status,
                   label=top["label"] if top else "Healthy Leaf",
                   confidence=top["confidence"] if top else 0.95,
                   recommendations=tips, detections=detections, annotated=annotated)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)