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

Before the game starts, 9 dots appear in a 3×3 grid across the screen. The player stares at each dot for ~2 seconds while the webcam captures frames. Each frame where the eyes are open becomes a labeled training sample:

```
(22 eye features) → (screen_x, screen_y)
```

After all 9 dots, roughly 500 labeled samples have been collected. The neural network trains on these samples immediately. The model is session-specific — it trains fresh every launch because head position and lighting change between sessions.

### 4. Neural Network

GazeNet is a small fully-connected feedforward neural network that learns the mapping from eye feature space to screen coordinate space.

**Architecture:**
```
Input (22)  →  Linear(22→64)  →  ReLU  →  Linear(64→32)  →  ReLU  →  Linear(32→2)  →  Output (x, y)
```

Each `Linear` layer computes `output = input × weights + bias`. The weight matrix for the first layer is shape (64, 22) — 64 neurons, each connected to all 22 inputs. This gives the model 64 different "views" of the eye feature vector to work with.

**Why ReLU activations?**

Without activation functions, stacking linear layers is mathematically equivalent to a single linear layer. ReLU (`f(x) = max(0, x)`) introduces non-linearity by clipping negative values to zero. This breaks the linearity and allows the model to learn curved decision boundaries.


**Training loop:**

```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1500):
    predictions = model(X_tensor)       # forward pass
    loss = criterion(predictions, Y_tensor)
    optimizer.zero_grad()               # clear stale gradients
    loss.backward()                     # backpropagation
    optimizer.step()                    # weight update
```

- **MSELoss** — Mean Squared Error: `mean((predicted_x - actual_x)² + (predicted_y - actual_y)²)`. Appropriate for continuous coordinate regression.
- **Adam optimizer** — adapts the learning rate per parameter using first and second moment estimates of the gradients. Converges faster than SGD on small datasets.
- **1500 epochs** — the entire dataset (~500 samples) passes through the network 1500 times. With a network this small and a dataset this size, this completes in under two seconds on CPU.
- **Backpropagation** — after each forward pass, PyTorch automatically differentiates the loss through every layer via the chain rule, computing the gradient of the loss with respect to each weight. The optimizer then nudges each weight in the direction that reduces the loss.



### 6. EMA Smoothing (`gaze/smoother.py`)

MediaPipe landmark detection has sub-pixel jitter that translates directly to cursor jitter. An **Exponential Moving Average** smooths the output:

```
x_smooth = α × x_raw + (1 − α) × x_prev
```

At α=0.5, the current frame has the same total weight as all previous frames combined. Older frames decay exponentially — nothing from more than ~7 frames ago meaningfully affects the output. On tracking loss (face leaves frame), the smoother resets so the cursor doesn't drag from a stale position.

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

---

## Stack

| Library | Role |
|---------|------|
| **MediaPipe** | Face landmark detection (FaceLandmarker Tasks API, 478 landmarks) |
| **PyTorch** | Neural network definition, training loop, inference |
| **OpenCV** | Webcam capture and frame preprocessing |
| **pygame** | Game loop, rendering, calibration UI |
| **NumPy** | Feature vector construction, array ops |




