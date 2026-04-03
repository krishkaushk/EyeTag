# EyeTag

An eye-tracking arcade shooter. Your gaze is the crosshair — look at enemies to aim, bullets fire automatically toward where you're looking.

## How it works

**Pipeline:**
1. Webcam frame → MediaPipe FaceLandmarker (478 landmarks)
2. Extract 22-feature vector: iris position relative to eye corners (×2 eyes), eye outline landmarks, Eye Aspect Ratio (EAR) values
3. Ridge regression maps features → screen (x, y)
4. KalmanSmoother filters jitter frame-to-frame
5. `atan2(gaze - center)` aims bullets from the ship at screen center

Calibration trains the regression model fresh on each launch/

## Calibration

9-point calibration: center → corners → edges. For each dot, ~2s of webcam frames are collected (blink frames discarded). Ridge regression is fit on the 500+ sample dataset.

**Tips:**
- Keep your head still — model assumes fixed head position
- Look at the center of each dot
- Good, even lighting helps landmark detection

## Setup

```bash
pip install -r requirements.txt
python -m game.main
```

Requires a webcam. Calibration runs automatically on launch.

## Stack

- **MediaPipe** — face landmark detection (FaceLandmarker Tasks API)
- **scikit-learn** — Ridge regression for gaze-to-screen mapping
- **eyetrax** — KalmanSmoother for gaze stabilization
- **OpenCV** — webcam capture
- **pygame** — game loop and rendering
