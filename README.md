# EyeTag

An eye-tracking arcade shooter. Look at enemies to aim, your gaze is the crosshair.

## How it works

Your webcam tracks your eye position in real time. A green crosshair follows your gaze, and bullets fire automatically from the center of the screen toward wherever you're looking. Enemies spawn from the edges and move toward you — shoot them before they reach the center.

Each wave gets faster and spawns enemies more frequently. Game over when an enemy reaches the center.

**Controls:** Eyes to aim — `R` to restart — `ESC` to quit

## Setup

```bash
pip install -r requirements.txt
python -m game.main
```

Requires a webcam. Tested on macOS.

## Calibration

Before the game starts, you'll go through eye calibration. The screen first detects your face and shows a countdown, then presents calibration targets to follow with your eyes while keeping your head still.

**9-point calibration** (default) — 9 fixed dots in a 3×3 grid, ~15 seconds total. You look at each dot while a progress ring fills, then the model trains. Fast and reliable.


Switch to Lissajous calibration (follow a moving dot) by commenting out the 9-point method and uncommenting the lissajous as seen in esimator.py:

```python
run_9_point_calibration(self.estimator)
# run_lissajous_calibration(self.estimator)
```

(9-point I find better)

**Tips for good calibration:**
- Keep your head still — the model assumes a fixed head position
- Look directly at the center of each dot, not around it
- Good lighting on your face helps MediaPipe detect landmarks reliably


## Kalman smoothing

Raw gaze predictions are noisy and eyes naturally tremble slightly even when holding still, and the landmark detector adds frame-to-frame variation. Without smoothing, the crosshair jitters constantly.


## Dependencies

- [eyetrax](https://github.com/ck-zhang/EyeTrax) — gaze estimation and calibration
- pygame — game loop and rendering
- OpenCV + MediaPipe — video capture and face landmarks
- scikit-learn — regression model for gaze prediction
