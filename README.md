# 😷 Face Mask Detection with Live Alert System

A real-time face mask detection system using **OpenCV**, **TensorFlow/Keras**, and **Flask**.  
Detects whether people are wearing face masks via webcam and fires live alerts when no mask is found.

---

## 📁 Project Structure

```
face_mask_detection/
├── app.py                  # Flask web application (live stream + upload)
├── train_model.py          # CNN training script
├── detect_realtime.py      # Standalone OpenCV detection script
├── generate_dataset.py     # Synthetic dataset generator (demo / offline)
├── demo_offline.py         # Offline demo video generator (no webcam needed)
├── requirements.txt        # Python dependencies
├── dataset/
│   ├── with_mask/          # Training images – masked faces
│   └── without_mask/       # Training images – unmasked faces
├── model/
│   ├── mask_detector.h5    # Trained Keras model (generated after training)
│   └── training_history.png
├── templates/
│   ├── index.html          # Live dashboard UI
│   └── upload.html         # Static-image test page
├── utils/
│   └── evaluate_model.py   # Confusion matrix + ROC curve
└── screenshots/            # Auto-saved screenshots (created at runtime)
```

---

## ⚙️ Setup

### 1. Clone / download the project

```bash
cd face_mask_detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🗂️ Dataset

### Option A – Kaggle Dataset (recommended for production)

1. Download the [Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) by Omkar Gurav
2. Unzip and place images in:
   - `dataset/with_mask/`
   - `dataset/without_mask/`

### Option B – Synthetic Dataset (for demo / offline testing)

```bash
python generate_dataset.py
```
Generates **500 synthetic images per class** (1,000 total) — no internet required.

---

## 🧠 Train the Model

```bash
python train_model.py
```

What it does:
- Loads images from `dataset/`
- Applies data augmentation (rotation, zoom, flip, shear)
- Trains a **4-block CNN** with BatchNorm + Dropout
- Saves best model to `model/mask_detector.h5`
- Plots accuracy / loss curves to `model/training_history.png`

**Expected results on Kaggle dataset:** ~98–99% validation accuracy.

---

## 🎥 Real-Time Detection (standalone)

```bash
python detect_realtime.py                        # default webcam
python detect_realtime.py --source 1             # second webcam
python detect_realtime.py --source video.mp4     # video file
python detect_realtime.py --threshold 0.6        # stricter threshold
```

**Controls:**
| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot |

---

## 🌐 Flask Web App

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### Features
| Route | Description |
|-------|-------------|
| `/` | Live dashboard with MJPEG stream + real-time stats |
| `/video_feed` | Raw MJPEG stream endpoint |
| `/upload` | Test with a static image file |
| `/stats` | JSON stats (faces, alerts, FPS, uptime) |
| `/health` | Health check / model status |

---

## 📊 Evaluate the Model

```bash
python utils/evaluate_model.py
```
Outputs:
- Classification report (precision / recall / F1 per class)
- Confusion matrix
- ROC curve with AUC score

Saved to `model/evaluation.png`.

---

## 🎬 Offline Demo (no webcam / model required)

```bash
python demo_offline.py --frames 60 --output demo_output.mp4
```
Generates a synthetic video demonstrating the detection overlay and alert system.

---

## 🏗️ Architecture

### CNN Model

```
Input (128×128×3)
    └─ Conv2D(32) → BN → MaxPool
    └─ Conv2D(64) → BN → MaxPool
    └─ Conv2D(128) → BN → MaxPool
    └─ Conv2D(128) → BN → GlobalAvgPool
    └─ Dense(256) + Dropout(0.5)
    └─ Dense(1, sigmoid)   ← binary: mask / no-mask
```

### Detection Pipeline

```
Webcam Frame
    └─ Haar Cascade → locate face ROIs
    └─ For each ROI:
        └─ Resize to 128×128
        └─ Normalise (÷255)
        └─ CNN predict → probability
        └─ Threshold → label + confidence
    └─ Draw bounding box + label
    └─ Alert if no-mask detected
```

---

## 🚨 Alert System

- **Visual**: Red banner overlaid on the video frame for 2 seconds
- **Console**: Timestamped log entry per alert
- **Flask dashboard**: Live JS-polled counter + banner animation
- **Screenshot**: Press `s` in standalone mode to capture frame

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `tensorflow / keras` | CNN model training & inference |
| `opencv-python` | Video capture, Haar Cascade, frame processing |
| `flask` | Web server & MJPEG streaming |
| `numpy` | Array ops |
| `Pillow` | Image I/O (dataset gen) |
| `scikit-learn` | Metrics, confusion matrix |
| `matplotlib` | Training plots |

---

## 📌 Notes

- The Haar Cascade (`haarcascade_frontalface_default.xml`) ships with OpenCV — no download needed.
- For GPU acceleration install `tensorflow-gpu` instead of `tensorflow`.
- In **demo mode** (no model loaded) predictions are random — train the model for real results.
- The Flask MJPEG stream uses one webcam process per server restart; for multi-user production use a proper WSGI server + shared frame buffer.

---

## 📄 License

MIT — free to use, modify, and distribute.
