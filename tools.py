import numpy as np
import cv2
import mediapipe as mp
from numpy import ndarray

hands_detector = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
hand_connections = mp.solutions.hands.HAND_CONNECTIONS
landmarks_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
connections_style = mp.solutions.drawing_styles.get_default_hand_connections_style()


def transform_for_RFC(landmarks):
    landmarks = convert_to_numpy(landmarks)
    landmarks_norm = normalize_landmarks(landmarks)
    return landmarks_norm.flatten()


def convert_to_numpy(landmarks):
    return np.array([[lm.x, lm.y] for lm in landmarks])


def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    middle_fingertip = landmarks[12]

    centered_landmarks = landmarks - wrist

    hand_size = np.linalg.norm(middle_fingertip - wrist)
    hand_size = max(hand_size, 1e-6)

    # Normalize landmarks by hand size
    normalized_landmarks: ndarray = centered_landmarks / hand_size

    # Scale to [0, 1] range
    # Find the min and max values along each axis
    min_values = np.min(normalized_landmarks, axis=0)
    max_values = np.max(normalized_landmarks, axis=0)

    scaled_landmarks = (normalized_landmarks - min_values) / (max_values - min_values + 1e-6)

    return scaled_landmarks


def draw_landmarks(landmarks, frame=None):
    if frame is None:
        frame = np.zeros((500, 500, 3), np.uint8)

    mp_draw.draw_landmarks(
        frame,
        landmarks,
        hand_connections,
        landmark_drawing_spec=landmarks_style,
        connection_drawing_spec=connections_style
    )
    return frame


def get_boundingbox(landmarks, displayed_frame):
    x = []
    y = []
    for lm in landmarks.landmark:
        h, w, c = displayed_frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        x.append(cx)
        y.append(cy)
    x1, x2 = min(x), max(x)
    y1, y2 = min(y), max(y)
    # add padding
    x1 -= 40
    x2 += 40
    y1 -= 40
    y2 += 40
    # make it square
    w = x2 - x1
    h = y2 - y1
    if w > h:
        d = w - h
        y1 -= d // 2
        y2 += d // 2
    else:
        d = h - w
        x1 -= d // 2
        x2 += d // 2

    return x1, y1, x2, y2


def draw_results(results, frame):
    # draw the points and connections
    hand_landmarks = results.multi_hand_landmarks[0]
    frame = draw_landmarks(hand_landmarks, frame)

    # draw the bounding box
    x1, y1, x2, y2 = get_boundingbox(hand_landmarks, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 0, 0), 2)

    # Calculate the position for the label's background rectangle
    label_height = 40  # Height of the blue rectangle for the label
    label_y1 = y1 - label_height if y1 - label_height > 0 else 0
    label_y2 = y1
    cv2.rectangle(frame, (x1, label_y1), (x2, label_y2), (255, 0, 0), -1)

    return frame, (x1, y1, x2, y2)