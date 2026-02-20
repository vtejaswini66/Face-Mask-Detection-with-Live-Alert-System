"""
app.py
------
Flask web application for face-mask detection.

Routes
------
GET  /              – live camera feed dashboard
GET  /video_feed    – MJPEG streaming endpoint
GET  /stats         – JSON stats (faces, masks, alerts)
POST /upload        – detect mask in an uploaded image
GET  /health        – health check

Run
---
    python app.py
    # then open http://localhost:5000
"""

import os
import sys
import io
import time
import datetime
import threading
import base64

import cv2
import numpy as np
from flask import (Flask, Response, render_template,
                   request, jsonify, redirect, url_for)

# ── Keras ─────────────────────────────────────────────────────────────────────
try:
    from tensorflow.keras.models import load_model as keras_load
    KERAS_OK = True
except ImportError:
    KERAS_OK = False

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 128
MODEL_PATH = "model/mask_detector.h5"
THRESHOLD  = 0.5
FONT       = cv2.FONT_HERSHEY_SIMPLEX

COLOUR_MASK    = (0,  200,   0)
COLOUR_NO_MASK = (0,    0, 220)
COLOUR_ALERT   = (0,    0, 255)

app = Flask(__name__)

# ── Global state (thread-safe enough for single-user demo) ────────────────────
_lock    = threading.Lock()
_stats   = {
    "total_faces"   : 0,
    "mask_count"    : 0,
    "no_mask_count" : 0,
    "alert_count"   : 0,
    "fps"           : 0.0,
    "uptime_s"      : 0,
    "started_at"    : datetime.datetime.now().isoformat(),
}
_alert_until = 0.0   # epoch seconds


# ── Model ─────────────────────────────────────────────────────────────────────
def _load_model():
    if KERAS_OK and os.path.isfile(MODEL_PATH):
        print(f"✅  Loading model …")
        return keras_load(MODEL_PATH)
    print("⚠️  Model not found – demo mode (random predictions).")
    return None

_model        = _load_model()
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _preprocess(roi):
    face = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return np.expand_dims(face.astype("float32") / 255.0, axis=0)


def _predict(roi):
    """Returns (label, confidence, has_mask)."""
    if _model is not None:
        pred     = float(_model.predict(_preprocess(roi), verbose=0)[0][0])
        has_mask = pred > THRESHOLD
        conf     = pred if has_mask else (1 - pred)
    else:
        has_mask = np.random.rand() > 0.4
        conf     = np.random.uniform(0.65, 0.99)
    label = "Mask" if has_mask else "No Mask"
    return label, conf, has_mask


def _annotate_frame(frame):
    global _alert_until
    h, w = frame.shape[:2]
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

    no_mask_now = 0
    with _lock:
        _stats["total_faces"] += len(faces)

    for (x, y, fw, fh) in faces:
        roi = frame[y:y+fh, x:x+fw]
        if roi.size == 0:
            continue
        label, conf, has_mask = _predict(roi)
        colour = COLOUR_MASK if has_mask else COLOUR_NO_MASK

        if not has_mask:
            no_mask_now += 1

        cv2.rectangle(frame, (x, y), (x+fw, y+fh), colour, 2)
        txt  = f"{label}: {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(txt, FONT, 0.6, 2)
        cv2.rectangle(frame, (x, y-th-10), (x+tw+6, y), colour, -1)
        cv2.putText(frame, txt, (x+3, y-5), FONT, 0.6, (255,255,255), 2)

    with _lock:
        _stats["mask_count"]    += (len(faces) - no_mask_now)
        _stats["no_mask_count"] += no_mask_now

    if no_mask_now > 0:
        _alert_until = time.time() + 2.0
        with _lock:
            _stats["alert_count"] += 1

    if time.time() < _alert_until:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), COLOUR_ALERT, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, "⚠  NO MASK – PLEASE WEAR A MASK!",
                    (20, 36), FONT, 0.7, (255,255,255), 2)

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w-110, h-12), FONT, 0.5, (180,180,180), 1)
    return frame


# ── MJPEG stream ──────────────────────────────────────────────────────────────
def _gen_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    t0      = time.time()
    fc      = 0
    fps     = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        fc += 1
        if fc % 15 == 0:
            fps = 15 / (time.time() - t0)
            t0  = time.time()
            with _lock:
                _stats["fps"]      = round(fps, 1)
                _stats["uptime_s"] += 1

        frame = _annotate_frame(frame)
        cv2.putText(frame, f"FPS {fps:.1f}", (10, 25),
                    FONT, 0.55, (200,200,200), 1)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    cap.release()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(_gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    with _lock:
        data = dict(_stats)
    return jsonify(data)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    buf   = np.frombuffer(file.read(), dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Cannot decode image"}), 400

    annotated = _annotate_frame(frame.copy())
    _, enc    = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64   = base64.b64encode(enc).decode("utf-8")

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    results = []
    for (x, y, fw, fh) in faces:
        roi = frame[y:y+fh, x:x+fw]
        if roi.size == 0:
            continue
        label, conf, _ = _predict(roi)
        results.append({"label": label, "confidence": round(conf, 3),
                         "bbox": [int(x), int(y), int(fw), int(fh)]})

    return jsonify({"image": img_b64, "results": results,
                    "face_count": len(faces)})


@app.route("/health")
def health():
    return jsonify({"status": "ok",
                    "model_loaded": _model is not None,
                    "keras": KERAS_OK})


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀  Starting Face Mask Detection server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
