# EyeTag

An eye-tracking arcade shooter. You look where you want to shoot and bullets fire automatically toward wherever you're looking.

```
Webcam → MediaPipe → Feature Extraction → Neural Network → EMA Smoother → Game
```

---

## How It Works

### 1. Face Landmark Detection

Every frame from the webcam is passed through **MediaPipe FaceLandmarker**, which returns 478 3D face landmarks in normalized coordinates. From those, only the eye region is used: iris centers (4 points each eye), eye outline corners, and the 6 landmarks used for blink detection.

### 2. Feature Extraction (`gaze/features.py`)

Raw landmark coordinates shift when your head moves, which would break any regression model trained on a fixed head position. The feature engineering step addresses this by computing **head-invariant features**:

- **Iris position relative to eye corners** — instead of using the raw iris x/y coordinates, the iris center is expressed as a fraction of the eye width and height. If your head moves left, both the iris and the eye corners shift together, so the ratio stays stable.
- **Eye outline landmarks** — 4 corner points per eye (8 total), providing coarse head orientation context.
- **Eye Aspect Ratio (EAR)** — a single number measuring how open each eye is: `EAR = (v1 + v2) / (2 * h)` where v1, v2 are vertical distances across the eye opening and h is the horizontal width. EAR < 0.21 reliably indicates a blink.

This produces a **22-dimensional feature vector** per frame.

Blink frames (EAR < 0.21) are discarded from both training data and live inference — a blink produces erratic iris positions that would corrupt the model.

### 3. Calibration (`gaze/calibration.py`)

Before the game starts, **25 dots appear in a 5×5 grid** across the screen. For each dot:

1. The dot appears and the player has **0.6 seconds** to move their eyes to it (nothing is collected yet — this prevents collecting frames where the eye is mid-travel)
2. A shrinking ring animates for **1 second** while frames are collected
3. Only frames where the eyes are open and the face is detected are kept

Each collected frame becomes a labeled training sample:

```
(22 eye features) → (screen_x, screen_y)
```

The 5×5 grid gives dense spatial coverage — the gaps between calibration points are small enough that the network can interpolate reliably. Total calibration time is ~40 seconds.

Targets are **normalized to 0–1** before training (divided by screen width/height). This keeps the output scale consistent with the input features, which are already in the 0–1 range from MediaPipe. Without this, MSE on raw pixel values (up to 1920/1080) creates an uneven loss landscape.

After training, an **accuracy map** is shown: each calibration point is drawn as a white dot, with a coloured arrow pointing to the model's mean prediction for that point. Green = accurate, red = large error. This lets you immediately see if specific regions trained poorly before starting the game.

### 4. Neural Network (`gaze/model.py`)

GazeNet is a deep fully-connected feedforward neural network mapping eye features to screen coordinates.

**Architecture:**
```
Input (22)
  → Linear(22 → 256) → GELU
  → Linear(256 → 256) → GELU
  → Linear(256 → 128) → GELU
  → Linear(128 → 64)  → GELU
  → Linear(64 → 32)   → GELU
  → Linear(32 → 2)
  → Output (x, y)  [normalized 0–1]
```

**Why GELU instead of ReLU?**

ReLU (`max(0, x)`) is piecewise linear — a network built from ReLU layers can only produce a function made of flat linear segments stitched together. As the gaze moves across the screen, the predicted coordinate moves in rigid straight lines between breakpoints.

GELU (`x · Φ(x)`, where Φ is the Gaussian CDF) is smooth and continuously differentiable everywhere. Stacking GELU layers produces a smooth curved function over the whole input space — the cursor flows naturally rather than snapping along straight edges. GELU is also used in GPT and BERT for the same reason.

**Why deeper?**

The 5-layer network (vs the original 2-layer) can learn more complex nonlinear relationships between iris position and screen coordinates. Each additional layer lets the model compose increasingly abstract representations of the eye state before mapping to a position.

**Training loop:**

```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1500):
    predictions = model(X_tensor)         # forward pass
    loss = criterion(predictions, Y_tensor)
    optimizer.zero_grad()                 # clear stale gradients
    loss.backward()                       # backpropagation
    optimizer.step()                      # weight update
```

- **MSELoss** — Mean Squared Error on normalized coordinates. Appropriate for continuous coordinate regression.
- **Adam optimizer** — adapts the learning rate per parameter using moment estimates of the gradients. Converges faster than SGD on small datasets.
- **Backpropagation** — PyTorch differentiates the loss through every layer via the chain rule, computing gradients for every weight. The optimizer nudges each weight in the direction that reduces the loss.

### 5. EMA Smoothing (`gaze/smoother.py`)

MediaPipe landmark detection has sub-pixel jitter that translates directly to cursor jitter. An **Exponential Moving Average** smooths the output:

```
x_smooth = α × x_raw + (1 − α) × x_prev
```

At α=0.35, each frame blends 35% new prediction with 65% history. Older frames decay exponentially — this reduces jitter from frame-to-frame noise without adding noticeable lag. On tracking loss (face leaves frame), the smoother resets so the cursor doesn't drag from a stale position.

Tunable via `ALPHA` at the top of `gaze/smoother.py`.

---

## Setup

```bash
pip install -r requirements.txt
python main.py
```

Requires a webcam. Tested on macOS.

**Calibration tips:**
- Keep your head still throughout
- Look at the center of each dot, not around it
- Even lighting improves landmark detection accuracy
- If accuracy looks poor on the map after calibration, press C during the game to recalibrate

---

## Stack

| Library | Role |
|---------|------|
| **MediaPipe** | Face landmark detection (FaceLandmarker Tasks API, 478 landmarks) |
| **PyTorch** | Neural network definition, training loop, inference |
| **OpenCV** | Webcam capture and frame preprocessing |
| **pygame** | Game loop, rendering, calibration UI |
| **NumPy** | Feature vector construction, array ops |
