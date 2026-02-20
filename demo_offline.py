"""
demo_offline.py
---------------
Runs the full detection pipeline on synthetic (generated) images without
requiring a webcam or trained model. Great for CI / demonstration.

Usage
-----
    python demo_offline.py [--frames 30] [--output demo_output.mp4]
"""

import argparse
import os
import sys
import time
import random

import cv2
import numpy as np

IMG_SIZE = 128
FONT     = cv2.FONT_HERSHEY_SIMPLEX

COLOUR_MASK    = (0,  200,   0)
COLOUR_NO_MASK = (0,    0, 220)
COLOUR_ALERT   = (0,    0, 255)
SKIN_TONES     = [
    (189, 224, 255), (145, 195, 240),
    ( 66, 134, 198), ( 36,  85, 141), (20,  60,  90)
]


def _draw_face(img, cx, cy, r, skin):
    cv2.ellipse(img, (cx, cy), (r, int(r*1.2)), 0, 0, 360, skin, -1)


def _draw_eyes(img, cx, cy, r):
    ew = max(int(r * 0.18), 4)
    for ex in [cx - int(r*0.3), cx + int(r*0.3)]:
        ey = cy - int(r*0.1)
        cv2.circle(img, (ex, ey), ew, (20, 30, 50), -1)


def _draw_mask(img, cx, cy, r, colour):
    x1, x2 = cx - int(r*0.85), cx + int(r*0.85)
    y1, y2 = cy + int(r*0.1),  cy + int(r*1.0)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (180,180,200), 2)
    for py in range(y1+6, y2-4, 8):
        cv2.line(img, (x1+4, py), (x2-4, py), (200,200,210), 1)


def _draw_mouth(img, cx, cy, r):
    axes  = (int(r*0.3), int(r*0.15))
    center= (cx, cy + int(r*0.55))
    cv2.ellipse(img, center, axes, 0, 0, 180, (80, 80, 180), 3)


def make_synthetic_frame(width=640, height=480, n_faces=2):
    frame = np.ones((height, width, 3), dtype=np.uint8) * 40   # dark bg
    # subtle grid
    for i in range(0, width, 40):
        cv2.line(frame, (i, 0), (i, height), (50,50,50), 1)
    for j in range(0, height, 40):
        cv2.line(frame, (0, j), (width, j), (50,50,50), 1)

    face_labels = []
    used_x = []
    for _ in range(n_faces):
        for _ in range(30):  # find non-overlapping position
            cx = random.randint(100, width - 100)
            if all(abs(cx - px) > 140 for px in used_x):
                break
        cy      = random.randint(160, height - 160)
        r       = random.randint(50, 70)
        skin    = random.choice(SKIN_TONES)
        has_mask= random.random() > 0.4
        used_x.append(cx)

        _draw_face(frame, cx, cy, r, skin[::-1])   # BGR
        _draw_eyes(frame, cx, cy, r)
        if has_mask:
            mask_col = random.choice([(200,130,0),(255,200,0),(200,200,200),(0,120,50)])
            _draw_mask(frame, cx, cy, r, mask_col)
        else:
            _draw_mouth(frame, cx, cy, r)

        face_labels.append((cx - r, cy - int(r*1.2),
                            2*r, int(r*2.4), has_mask))
    return frame, face_labels


def annotate(frame, face_labels, alert_active):
    for (x, y, w, h, has_mask) in face_labels:
        col    = COLOUR_MASK if has_mask else COLOUR_NO_MASK
        label  = "Mask" if has_mask else "No Mask"
        conf   = random.uniform(0.82, 0.99)

        cv2.rectangle(frame, (x, y), (x+w, y+h), col, 2)
        txt = f"{label}: {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(txt, FONT, 0.55, 2)
        cv2.rectangle(frame, (x, y-th-10), (x+tw+6, y), col, -1)
        cv2.putText(frame, txt, (x+3, y-5), FONT, 0.55, (255,255,255), 2)

    if alert_active:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (frame.shape[1], 50), COLOUR_ALERT, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, "WARNING: NO MASK DETECTED – PLEASE WEAR A MASK",
                    (20, 33), FONT, 0.65, (255,255,255), 2)
    return frame


def run(n_frames, output_path, show):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw     = cv2.VideoWriter(output_path, fourcc, 10, (640, 480))

    alert_until = 0.0
    for fi in range(n_frames):
        n       = random.randint(1, 3)
        frame, labels = make_synthetic_frame(n_faces=n)

        no_mask = sum(1 for *_, hm in labels if not hm)
        if no_mask:
            alert_until = time.time() + 2.0

        frame = annotate(frame, labels, time.time() < alert_until)

        # HUD
        cv2.putText(frame, f"Frame {fi+1}/{n_frames}", (10,24),
                    FONT, 0.55, (180,180,180), 1)
        cv2.putText(frame, "DEMO MODE", (10, 470),
                    FONT, 0.5, (80,80,80), 1)

        vw.write(frame)
        if show:
            cv2.imshow("Demo – Face Mask Detection", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        else:
            sys.stdout.write(f"\r  Writing frame {fi+1}/{n_frames} …")
            sys.stdout.flush()

    vw.release()
    if show:
        cv2.destroyAllWindows()
    print(f"\n✅  Demo video saved → {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames",  default=60,   type=int)
    ap.add_argument("--output",  default="demo_output.mp4")
    ap.add_argument("--show",    action="store_true",
                    help="Display preview window (requires display)")
    args = ap.parse_args()
    run(args.frames, args.output, args.show)
