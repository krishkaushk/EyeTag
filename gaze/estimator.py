# Estimates gaze for eye tracking with eyetrax

import torch
import cv2
from gaze.features import extract_features
from gaze.calibration import Calibrator
from gaze.smoother import EMAsmoother, ALPHA
from game.config import SCREEN_WIDTH, SCREEN_HEIGHT

class GazeTracker:
    def __init__(self, alpha=ALPHA):
        self.cap = cv2.VideoCapture(0)
        self.model = None
        self.last_x = None
        self.last_y = None
        self.raw_y = None
        self._smoother = EMAsmoother(alpha)


    def calibrate(self, screen):
        calibrator = Calibrator(screen, self.cap)
        self.model = calibrator.run()
        self.calibrated = True

    def get_coords(self):
        if self.model is None:
            return None

        ret, frame = self.cap.read()

        if not ret:
            return None

        try:
            features, is_blinking = extract_features(frame)

            
            if features is None or is_blinking:
                self._smoother.reset()
                return self._last_known()

            # numpy (22,) → tensor (1, 22): unsqueeze adds the batch dimension
            feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

            # no_grad: skip gradient tracking — we're predicting, not training
            with torch.no_grad():
                output = self.model(feature_tensor)  # tensor (1, 2)

            # tensor (1, 2) → numpy, then scale back up from 0-1 → pixel coords
            prediction = output.numpy()
            x = int(prediction[0][0] * SCREEN_WIDTH)
            y = int(prediction[0][1] * SCREEN_HEIGHT)

            x = max(0, min(x, SCREEN_WIDTH))
            y = max(0, min(y, SCREEN_HEIGHT))

            self.raw_y = y
            x, y = self._smoother.update(x, y)

            print(f"RAW Y: {self.raw_y}  |  SMO Y: {y}")

            self.last_x = x
            self.last_y = y

            return (x, y)

        except Exception as e:
            print(f"Prediction error: {e}")
            return self._last_known()
    
    def _last_known(self):
        if self.last_x is not None and self.last_y is not None:
            return (self.last_x, self.last_y)
        return None



    def release(self):
        self.cap.release()