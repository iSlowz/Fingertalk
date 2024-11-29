import pickle
import cv2
from collections import deque
from tools import hands_detector, draw_results, transform_for_RFC


if __name__ == '__main__':
    model = pickle.load(open('random_forest_model.pkl', 'rb'))
    idx_to_label = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "J",
        10: "K",
        11: "L",
        12: "M",
        13: "N",
        14: "O",
        15: "P",
        16: "Q",
        17: "R",
        18: "S",
        19: "Space",
        20: "T",
        21: "U",
        22: "V",
        23: "W",
        24: "X",
        25: "Y",
        26: "Z"
    }
    fps = 15
    len_deque = max(5, int(1.5*fps))
    last_letters = deque(maxlen=len_deque)
    last_confs = deque(maxlen=len_deque)

    cap = cv2.VideoCapture(0)
    num_frame_skip = max(1, int(int(cap.get(cv2.CAP_PROP_FPS)) / fps))
    prediction = None
    while cap.isOpened():
        for i in range(num_frame_skip):
            ret, frame = cap.read()
            if not ret:
                break  # Exit if the video ends

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
            label = idx_to_label[prediction]

            last_letters.append(label)
            last_confs.append(conf)
            if len(set(last_letters)) == 1 and len(last_letters) == last_letters.maxlen:
                avg_conf = sum(last_confs) / len(last_confs)
                if avg_conf > 0.5:
                    print(label, avg_conf)
                    last_letters.clear()

            text = f'{label} : {round(conf * 100, 1)}'
            cv2.putText(displayed_frame, text, (boundingbox[0]+10, boundingbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Hand Detection (press q to exit)", displayed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
