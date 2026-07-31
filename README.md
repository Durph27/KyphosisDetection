# Kyphosis Detection

Real-time posture monitoring desktop app. Uses your webcam + MediaPipe Pose to detect slouching and alert you.

## Setup

```bash
python -m venv kyphosisdetection
.\kyphosisdetection\Scripts\Activate.ps1   # Windows
pip install opencv-python mediapipe PySide6
```

The pose model (`pose_landmarker_full.task`) downloads automatically on first run.

## Run

```bash
python application.py
```

## How It Works

1. **Calibrate** — Sit straight for 3 seconds. The app records your baseline shoulder position, nose-to-shoulder distance, shoulder width, and face width.
2. **Monitor** — Each frame is compared against the baseline using four metrics:
   - **Shoulder drop** — how far shoulders have fallen (forward hunch)
   - **Nose ratio** — nose-to-shoulder distance shrinkage (head drop)
   - **Shoulder/face width ratios** — projected size decrease (leaning backward)
3. **Alert** — If bad posture persists beyond a configurable delay, the app triggers an alert window, optional screen blur, and optional audio.

## Controls

| Button | Action |
|---|---|
| Recalibrate | Re-record baseline (sit straight) |
| Settings | Configure alert delay, sound, screen blur |
| Hide to Background | Minimize to system tray |
| Threshold slider | Adjust detection sensitivity (0–100%) |
