import cv2
import mediapipe as mp
from PIL import Image
from aprentissage.test_model import predict_class


def get_boundingbox(landmarks):
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


def draw(results, displayed_frame):
    # draw the points and connections
    hand_landmarks = results.multi_hand_landmarks[0]
    mp_draw.draw_landmarks(displayed_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # draw the bounding box
    x1, y1, x2, y2 = get_boundingbox(hand_landmarks)
    cv2.rectangle(displayed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # tell the user which hand is detected left or right
    if results.multi_handedness:
        handedness = results.multi_handedness[0].classification[0].label
        confidence_handedness = results.multi_handedness[0].classification[0].score
        text = f"{handedness} ({confidence_handedness:.2f})"
        cv2.putText(displayed_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    return displayed_frame, (x1, y1, x2, y2)


if __name__ == '__main__':
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    start_time = cv2.getTickCount()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for natural interaction
        frame = cv2.flip(frame, 1)
        displayed_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            displayed_frame, boundingbox = draw(results, displayed_frame)

        # show the hand detected in a separate window every 1 seconds
        if cv2.getTickCount() - start_time > 1 * cv2.getTickFrequency():
            if results.multi_hand_landmarks:
                x1, y1, x2, y2 = boundingbox
                hand = frame[y1:y2, x1:x2]
                if hand.shape[0] and hand.shape[1]:
                    PIL_hand = Image.fromarray(cv2.cvtColor(hand, cv2.COLOR_BGR2RGB))
                    classe = predict_class(PIL_hand)
                    cv2.putText(hand, classe, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                    cv2.imshow('hand', cv2.resize(hand, (400, 400)))
            start_time = cv2.getTickCount()

        cv2.imshow("Hand Detection (press q to exit)", displayed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
