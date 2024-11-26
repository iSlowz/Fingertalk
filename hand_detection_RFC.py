import pickle
import json
import cv2
from tools import hands_detector, draw_results, transform_for_RFC


if __name__ == '__main__':
    model = pickle.load(open('random_forest_model.pkl', 'rb'))
    idx_to_label = json.load(open('idx_2_label.json', 'r'))
    idx_to_label = {int(k): v for k, v in idx_to_label.items()}

    fps = 30
    cap = cv2.VideoCapture(0)
    num_frame_skip = max(1, int(int(cap.get(cv2.CAP_PROP_FPS)) / fps))
    prediction = None
    while cap.isOpened():
        for i in range(num_frame_skip):
            ret, frame = cap.read()
            if not ret:
                break  # Exit if the video ends

        if not ret:
            break

        # Flip frame for natural interaction
        frame = cv2.flip(frame, 1)
        displayed_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands_detector.process(rgb_frame)

        if results.multi_hand_landmarks:
            displayed_frame, boundingbox = draw_results(results, displayed_frame)
            landmarks = results.multi_hand_landmarks[0].landmark
            sample = transform_for_RFC(landmarks)
            prediction = model.predict([sample])[0]
            conf = model.predict_proba([sample])[0][model.classes_ == prediction][0]
            conf = round(conf * 100, 1)
            label = idx_to_label[prediction]
            cv2.putText(displayed_frame, f'{label} : {conf}',
                        (boundingbox[0], boundingbox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (150, 0, 0), 1)

        cv2.imshow("Hand Detection (press q to exit)", displayed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        start_tick = cv2.getTickCount()

    cap.release()
    cv2.destroyAllWindows()
