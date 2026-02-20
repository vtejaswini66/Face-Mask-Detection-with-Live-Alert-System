"""
detect_realtime.py
------------------
Real-time face-mask detection using:
  • OpenCV Haar Cascade  – face localisation
  • Keras CNN model      – mask / no-mask classification
  • Alert system        – visual + console alert when mask is absent

Usage
-----
    python detect_realtime.py [--source 0] [--model model/mask_detector.h5]

Arguments
---------
    --source   int or path  webcam index (0) or video file path   (default: 0)
    --model    str          path to trained .h5 model             (default: model/mask_detector.h5)
    --threshold float       confidence threshold 0-1              (default: 0.5)
    --alert-sound           play beep alert (requires 'playsound')(default: off)

Keys
----
    q  –  quit
    s  –  screenshot
"""

import argparse
import os
import sys
import time
import datetime

import cv2
import numpy as np

# ── Keras ─────────────────────────────────────────────────────────────────────
try:
    from tensorflow.keras.models import load_model
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️  TensorFlow not found – running in DEMO mode (random predictions).")

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 128
ALERT_HOLD  = 2.0      # seconds to hold alert colour on screen
FONT        = cv2.FONT_HERSHEY_SIMPLEX

COLOUR_MASK   = (0,  200,  0)    # green  – mask detected
COLOUR_NO_MASK= (0,   0, 200)    # red    – no mask
COLOUR_ALERT  = (0,   0, 255)    # bright red for alert bar

# ── Alert state ───────────────────────────────────────────────────────────────
class AlertSystem:
    def __init__(self):
        self._last_alert   = 0.0
        self._alert_count  = 0
        self._total_alerts = 0

    def trigger(self):
        now = time.time()
        if now - self._last_alert > ALERT_HOLD:
            self._alert_count  += 1
            self._total_alerts += 1
            self._last_alert    = now
            print(f"🚨  ALERT [{datetime.datetime.now():%H:%M:%S}]  NO MASK DETECTED  "
                  f"(total alerts: {self._total_alerts})")

    @property
    def active(self):
        return time.time() - self._last_alert < ALERT_HOLD


# ── Pre-processing ────────────────────────────────────────────────────────────
def preprocess_face(face_roi: np.ndarray) -> np.ndarray:
    """Resize → normalise → add batch dim."""
    face = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)


# ── Overlay helpers ──────────────────────────────────────────────────────────
def draw_face_box(frame, x, y, w, h, label, confidence, colour):
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
    text = f"{label}: {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), colour, -1)
    cv2.putText(frame, text, (x + 3, y - 5), FONT, 0.6, (255, 255, 255), 2)


def draw_alert_bar(frame, text="⚠  NO MASK DETECTED – PLEASE WEAR A MASK!"):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), COLOUR_ALERT, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (20, 34), FONT, 0.75, (255, 255, 255), 2)


def draw_stats(frame, fps, face_count, no_mask_count):
    h, w = frame.shape[:2]
    info = [
        f"FPS: {fps:.1f}",
        f"Faces: {face_count}",
        f"No-mask: {no_mask_count}",
        datetime.datetime.now().strftime("%H:%M:%S"),
    ]
    for i, line in enumerate(info):
        cv2.putText(frame, line, (w - 160, 25 + i * 22),
                    FONT, 0.55, (200, 200, 200), 1)


# ── Main ──────────────────────────────────────────────────────────────────────
def run(source, model_path, threshold, save_screenshots):
    # ── Load model ────────────────────────────────────────────────────────────
    model = None
    if KERAS_AVAILABLE and os.path.isfile(model_path):
        print(f"✅  Loading model from {model_path} …")
        model = load_model(model_path)
        print("✅  Model loaded.")
    elif KERAS_AVAILABLE:
        print(f"⚠️  Model not found at '{model_path}'. Run train_model.py first.")
        print("    Continuing in DEMO mode …")
    
    # ── Load Haar Cascade ─────────────────────────────────────────────────────
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade  = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        sys.exit("❌  Could not load Haar Cascade. Check OpenCV installation.")
    print("✅  Haar Cascade loaded.")

    # ── Open video source ─────────────────────────────────────────────────────
    try:
        src = int(source)
    except ValueError:
        src = source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open video source: {source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    alert_sys   = AlertSystem()
    frame_count = 0
    fps_timer   = time.time()
    fps         = 0.0
    ss_dir      = "screenshots"

    print("\n▶  Detection running – press [q] to quit, [s] for screenshot.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended.")
            break

        frame_count += 1

        # FPS calc every 30 frames
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_timer)
            fps_timer = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors= 5,
            minSize     = (60, 60),
        )

        no_mask_count = 0

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            if model is not None:
                blob       = preprocess_face(roi)
                pred       = float(model.predict(blob, verbose=0)[0][0])
                # class_indices depends on alphabetical order of folders:
                #   with_mask = 1, without_mask = 0  → pred > 0.5 = with_mask
                has_mask   = pred > threshold
                confidence = pred if has_mask else (1 - pred)
            else:
                # DEMO: random prediction
                has_mask   = np.random.rand() > 0.4
                confidence = np.random.uniform(0.6, 0.99)

            if has_mask:
                label  = "Mask"
                colour = COLOUR_MASK
            else:
                label        = "No Mask"
                colour       = COLOUR_NO_MASK
                no_mask_count += 1
                alert_sys.trigger()

            draw_face_box(frame, x, y, w, h, label, confidence, colour)

        # ── Alert overlay ────────────────────────────────────────────────────
        if alert_sys.active:
            draw_alert_bar(frame)

        # ── Stats overlay ────────────────────────────────────────────────────
        draw_stats(frame, fps, len(faces), no_mask_count)

        # ── Title bar colour ──────────────────────────────────────────────────
        title = "Face Mask Detection  |  q=quit  s=screenshot"
        cv2.imshow(title, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            os.makedirs(ss_dir, exist_ok=True)
            ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pth = os.path.join(ss_dir, f"screenshot_{ts}.jpg")
            cv2.imwrite(pth, frame)
            print(f"📸  Screenshot saved → {pth}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n🏁  Session ended.  Total frames: {frame_count}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Face Mask Detection – Real-Time")
    ap.add_argument("--source",      default="0",
                    help="Webcam index (0) or video file path")
    ap.add_argument("--model",       default="model/mask_detector.h5",
                    help="Path to trained Keras model")
    ap.add_argument("--threshold",   default=0.5, type=float,
                    help="Confidence threshold (default 0.5)")
    ap.add_argument("--screenshots", action="store_true",
                    help="Enable auto-screenshots on alert")
    args = ap.parse_args()

    run(args.source, args.model, args.threshold, args.screenshots)
