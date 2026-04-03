# Estimates gaze for eye tracking with eyetrax

import cv2
from eyetrax import GazeEstimator
from eyetrax.calibration.nine_point import run_9_point_calibration
from eyetrax.calibration.lissajous import run_lissajous_calibration  # alternative
from eyetrax.filters import KalmanSmoother

class GazeTracker:
    def __init__(self):
        self.estimator = GazeEstimator()
        self.cap = cv2.VideoCapture(0)
        self.last_x = None
        self.last_y = None
        self.calibrated = False
        self._smoother = KalmanSmoother()

    def calibrate(self):
        # Pre-anchor the window at the top-left corner before eyetrax creates it,
        # preventing the slight rightward offset on macOS.
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Calibration", 0, 0)
        run_9_point_calibration(self.estimator)
        # run_lissajous_calibration(self.estimator)  # swap in for smoother coverage
        self.calibrated = True

    def get_coords(self):
        if not self.calibrated:
            return None

        ret, frame = self.cap.read()

        if not ret:
            return None

        try:
            features, blink = self.estimator.extract_features(frame)

            if features is None or blink:
                return self._last_known()

            x, y = self.estimator.predict([features])[0]

            sx, sy = self._smoother.step(int(x), int(y))
            self.last_x = sx
            self.last_y = sy

            return (self.last_x, self.last_y)

        except Exception:
            return self._last_known()
        
    
    def _last_known(self):
        if self.last_x is not None and self.last_y is not None:
            return (self.last_x, self.last_y)
        return None
    

    def save(self, path="gaze_model.pkl"):
        self.estimator.save_model(path)

    def load(self, path="gaze_model.pkl"):
        try:
            self.estimator.load_model(path)
            self.calibrated = True
        except Exception:
            print("No saved calibration found - please calibrate")


    def release(self):
        self.cap.release()