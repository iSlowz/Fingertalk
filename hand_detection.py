import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
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

    # Draw hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(displayed_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # get the bounding box of the hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x = []
            y = []
            for lm in hand_landmarks.landmark:
                h, w, c = displayed_frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x.append(cx)
                y.append(cy)
            x1, x2 = min(x), max(x)
            y1, y2 = min(y), max(y)
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
            cv2.rectangle(displayed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # if there are two hands detected, warn the user
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        cv2.putText(displayed_frame, "show only one hand", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # continue

    # show the hand detected in a separate window every 2 seconds
    if cv2.getTickCount() - start_time > 2 * cv2.getTickFrequency():
        if results.multi_hand_landmarks:
            hand = frame[y1:y2, x1:x2]
            # @Maxime tu peux extraire la main ici
            if hand.shape[0] and hand.shape[1]:
                cv2.imshow("Hand", cv2.resize(hand, (400, 400)))
        start_time = cv2.getTickCount()

    cv2.imshow("Hand Detection", displayed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
