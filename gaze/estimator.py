# Estimates gaze for eye tracking with eyetrax

import cv2
from gaze.features import extract_features
from gaze.calibration import Calibrator
from eyetrax.filters import KalmanSmoother
from game.config import SCREEN_WIDTH, SCREEN_HEIGHT

class GazeTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.model = None
        self.last_x = None
        self.last_y = None
        self._smoother = KalmanSmoother()

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
                return self._last_known()

            prediction = self.model.predict([features])
            x = int(prediction[0][0])
            y = int(prediction[0][1])

            x = max(0, min(x, SCREEN_WIDTH))
            y = max(0, min(y, SCREEN_HEIGHT))

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