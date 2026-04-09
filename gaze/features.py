# gets eye features from mediapipe

import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from pathlib import Path


BLINK_THRESHOLD = 0.21

_TASK_PATH = Path.home() / ".cache" / "eyetrax" / "mediapipe" / "face_landmarker.task"
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not _TASK_PATH.exists():
    print("Downloading face landmark model (~29MB, one time only)...")
    _TASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_MODEL_URL, _TASK_PATH)
    print("Download complete.")

_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(_TASK_PATH)),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
face_mesh = vision.FaceLandmarker.create_from_options(_options)
_last_ts_ms = 0


# --- Landmark indices ---
# MediaPipe gives 468 points. These are the specific ones we care about.
# Left eye outline
LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# Right eye outline
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# Iris centers (only available when refine_landmarks=True)
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
# Eye-Aspect-Ratio landmarks - the 6 points used to calculate eye openness
LEFT_EAR_POINTS  = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_POINTS = [33,  160, 158, 133, 153, 144]




def calculate_ear(landmarks, eye_points, image_w, image_h):
    # EAR = Eye Aspect Ratio
    # Measures how open the eye is as a single number
    # Below ~0.21 = blinking

    # Get the 6 landmark coordinates for this eye
    coords = []
    for idx in eye_points:
        lm = landmarks[idx]
        coords.append((lm.x * image_w, lm.y * image_h))

    # Vertical distances (how tall the eye opening is)
    v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))

    # Horizontal distance (how wide the eye is)
    h  = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))

    # The ratio - small when eye is closed, larger when open
    ear = (v1 + v2) / (2.0 * h)
    return ear


def get_relative_iris(landmarks, iris_points, eye_points):
    # Calculate iris position RELATIVE to eye corners
    # This stays consistent even when head moves

    # Get iris center — average of the 4 iris landmark points
    iris_x = np.mean([landmarks[i].x for i in iris_points])
    iris_y = np.mean([landmarks[i].y for i in iris_points])

    # Get eye corners
    # First point in eye_points = left corner
    # Last point in eye_points  = right corner (index 8 in the 16-point list)
    left_corner_x  = landmarks[eye_points[0]].x
    left_corner_y  = landmarks[eye_points[0]].y
    right_corner_x = landmarks[eye_points[8]].x
    right_corner_y = landmarks[eye_points[8]].y

    # Eye width and height for normalization
    eye_width  = right_corner_x - left_corner_x
    eye_height = abs(landmarks[eye_points[4]].y - landmarks[eye_points[12]].y)

    # Guard against division by zero if eye width is somehow 0
    if eye_width < 0.001 or eye_height < 0.001:
        return None

    # Relative position — 0.5 means iris is centered in eye
    rel_x = (iris_x - left_corner_x) / eye_width
    rel_y = (iris_y - left_corner_y) / eye_height

    return rel_x, rel_y



def extract_features(frame):
    global _last_ts_ms
    # Takes a raw webcam frame
    # Returns (feature_vector, is_blinking) or (None, False) if no face found

    image_h, image_w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    ts_ms = int(time.time() * 1000)
    if ts_ms <= _last_ts_ms:
        ts_ms = _last_ts_ms + 1
    _last_ts_ms = ts_ms

    result = face_mesh.detect_for_video(mp_image, ts_ms)

    if not result.face_landmarks:
        return None, False

    landmarks = result.face_landmarks[0]

    left_ear  = calculate_ear(landmarks, LEFT_EAR_POINTS,  image_w, image_h)
    right_ear = calculate_ear(landmarks, RIGHT_EAR_POINTS, image_w, image_h)
    avg_ear   = (left_ear + right_ear) / 2.0
    is_blinking = avg_ear < BLINK_THRESHOLD

    features = []

    left_iris_rel  = get_relative_iris(landmarks, LEFT_IRIS,  LEFT_EYE)
    right_iris_rel = get_relative_iris(landmarks, RIGHT_IRIS, RIGHT_EYE)

    # If either eye calculation failed, bail out
    if left_iris_rel is None or right_iris_rel is None:
        return None, False

    features.append(left_iris_rel[0])   # left iris relative x
    features.append(left_iris_rel[1])   # left iris relative y
    features.append(right_iris_rel[0])  # right iris relative x
    features.append(right_iris_rel[1])  # right iris relative y

    for idx in LEFT_EYE[:4] + RIGHT_EYE[:4]:
        lm = landmarks[idx]
        features.append(lm.x)
        features.append(lm.y)

    features.append(left_ear)
    features.append(right_ear)

    return np.array(features), is_blinking
