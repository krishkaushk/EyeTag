# Estimates gaze for eye tracking with eyetrax

import cv2
from eyetrax import GazeEstimator

class GazeTracker:
    def __init__(self):
        self.estimator = GazeEstimator()
        self.cap = cv2.VideoCapture(0)
        self.last_x = None
        self.last_y = None
        self.calibrated = False

    def calibrate(self):
        from eyetrax import run_9_point_calibration
        # Pre-anchor the window at the top-left corner before eyetrax creates it,
        # preventing the slight rightward offset on macOS.
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Calibration", 0, 0)
        run_9_point_calibration(self.estimator)
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

            self.last_x = int(x)
            self.last_y = int(y)

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