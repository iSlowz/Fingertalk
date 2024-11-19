import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd


def get_landmarks(image_path: str) -> np.ndarray:
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Read an image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image
    results = hands.process(img_rgb)

    # Extract landmarks
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            return hand_landmarks
        else:
            print(f'Image {image_path} contains more than one hand')
    else:
        print(f'No hands found in image {image_path}')

    return False


def landmarks_to_feature_vector(landmarks, additional_features: bool = False) -> np.ndarray:
    # Convert landmarks to a list of (x, y, z) tuples
    landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]

    # Normalize based on wrist and middle finger
    wrist = np.array(landmarks[0])
    middle_tip = np.array(landmarks[9])
    scale = np.linalg.norm(middle_tip - wrist)

    normalized_landmarks = [(lm - wrist) / scale for lm in landmarks]

    # Flatten the coordinates
    feature_vector = np.array(normalized_landmarks).flatten()

    if additional_features:
        # Additional features: distances between thumb tip and other fingertips
        thumb_tip = normalized_landmarks[4]
        finger_tips = [normalized_landmarks[i] for i in [8, 12, 16, 20]]

        for fingertip in finger_tips:
            distance = np.linalg.norm(thumb_tip - fingertip)
            feature_vector = np.append(feature_vector, distance)

    return feature_vector


def images_2_landmarks(dir_path = 'ASL_dataset/asl_alphabet_train/asl_alphabet_train'):
    rows = []
    errors = []

    dossiers = sorted(os.listdir(dir_path))
    for dossier in dossiers:
        print(f'Processing folder {dossier}')
        dossier_chemin = os.path.join(dir_path, dossier)
        if os.path.isdir(dossier_chemin):
            for image in os.listdir(dossier_chemin):
                if image.endswith('.jpg'):
                    landmark = get_landmarks(os.path.join(dossier_chemin, image))
                    if landmark:
                        feature_vector = landmarks_to_feature_vector(landmark)
                        new_row = {'feature_vector': feature_vector, 'label': dossier}
                        rows.append(new_row)
                    else:
                        print(f'No landmarks found for image {image}')
                        errors.append(image)

    print(f'Errors: {errors}')
    print(f'no landmarks found in {len(errors)} images')
    df = pd.DataFrame(rows)
    df.to_csv('landmarks_dataset/landmarks_train.csv', index=False)


def test_1_img(image_path = 'ASL_dataset/asl_alphabet_train/asl_alphabet_train/A/A1.jpg'):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Image', img)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    # Process the image
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Landmarks', img)
    cv2.waitKey(0)

    landmarks = get_landmarks(image_path)
    feature_vector = landmarks_to_feature_vector(landmarks, additional_features=True)
    print(feature_vector)


if __name__ == '__main__':
    test_1_img(image_path='ASL_dataset/asl_alphabet_train/asl_alphabet_train/A/A106.jpg')

    # images_2_landmarks()
    # print('Done')
