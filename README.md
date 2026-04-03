# EyeTag

An eye-tracking arcade shooter. Your gaze is the crosshair — look at enemies to aim, bullets fire automatically toward wherever you're looking.

## How it works

**Pipeline:**
1. Webcam frame → MediaPipe FaceLandmarker (478 face landmarks)
2. Extract 22-feature vector per frame: iris position relative to eye corners (×2), eye outline landmarks, Eye Aspect Ratio (EAR)
3. Ridge regression maps features → screen (x, y)
4. KalmanSmoother filters frame-to-frame jitter
5. `atan2(gaze − center)` aims bullets from the ship at screen center

## ML

Supervised regression. During calibration, you stare at 9 known points on screen each frame becomes a labeled training sample: 22 eye features → (screen_x, screen_y). After ~500 samples, Ridge regression fits a linear map from eye space to screen space.

A few design choices that make a linear model work:

- **Iris relative to eye corners** — iris position is normalized by eye corner coordinates, making it partially invariant to head movement. The raw eye outline landmarks in the feature vector aren't normalized, so large head movements will drift predictions — small movements are fine
- **Ridge regularization (α=1.0)** — 500 samples and 22 features is small enough that plain linear regression would overfit; Ridge keeps it stable
- **Blink frames discarded** — EAR < 0.21 flags a blink; those frames are dropped from training to keep labels clean

Inference is a single matrix multiply, fast enough to run every frame.

## Calibration

9 dots in a 3×3 grid, top-left to bottom-right. Stare at each for ~2s while the progress ring fills. Blinks are ignored. Model trains on completion (~15s total).


**Tips:**
- Keep your head still
- Look at the center of each dot, not around it
- Even lighting helps landmark detection

## Setup

```bash
pip install -r requirements.txt
python -m game.main
```

Requires a webcam. Tested on macOS.

## Stack

- **MediaPipe** — face landmark detection (FaceLandmarker Tasks API)
- **scikit-learn** — Ridge regression for gaze-to-screen mapping
- **eyetrax** — KalmanSmoother for gaze stabilization
- **OpenCV** — webcam capture
- **pygame** — game loop and rendering
