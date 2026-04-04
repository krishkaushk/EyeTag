# EyeTag

An eye-tracking arcade shooter. Your gaze is the crosshair — look at enemies to aim, bullets fire automatically toward wherever you're looking.

## How it works

**Pipeline:**
1. Webcam frame → MediaPipe FaceLandmarker (478 face landmarks)
2. Extract 22-feature vector per frame: iris position relative to eye corners (×2), eye outline landmarks, Eye Aspect Ratio (EAR)
3. Ridge regression maps features → screen (x, y)
4. EMA smoother (α=0.5) filters frame-to-frame jitter
5. `atan2(gaze − center)` aims bullets from the ship at screen center

## ML

Supervised regression. During calibration, you stare at 9 known points on screen each frame becomes a labeled training sample: 22 eye features → (screen_x, screen_y). After ~500 samples, Ridge regression fits a linear map from eye space to screen space.

A few design choices that make a linear model work:

- **Iris relative to eye corners** — iris position is normalized by eye corner coordinates, making it partially invariant to head movement. The raw eye outline landmarks in the feature vector aren't normalized, so large head movements will drift predictions — small movements are fine
- **Ridge regularization (α=0.2)** — 500 samples and 22 features is small enough that plain linear regression would overfit; Ridge keeps it stable
- **Blink frames discarded** — EAR < 0.21 flags a blink; those frames are dropped from training to keep labels clean

Inference is a single matrix multiply, fast enough to run every frame.

## Smoothing

Raw gaze landmark detection has sub-pixel jitter that translates to cursor jitter. An **Exponential Moving Average (EMA)** smooths the output:

```
x_smooth = α * x_raw + (1 − α) * x_prev
```

Each new prediction is blended with the previous smoothed position. 

α=0.5 means more weight on history than the new measurement, giving a smoother but slightly laggier cursor. To change, set `ALPHA` at the top of `gaze/smoother.py`.

On tracking loss (e.g. face leaves frame), the smoother resets so it doesn't drag the cursor from a stale position when tracking resumes.

## Calibration

9 dots in a 3×3 grid, top-left to bottom-right. Stare at each for ~2s while the progress ring fills. Blinks are ignored. Model trains on completion (~25s total).


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
- **OpenCV** — webcam capture
- **pygame** — game loop and rendering
- **numpy** — feature vector construction and matrix ops
- **Custom EMA smoother** (`gaze/smoother.py`) — no external smoothing library
