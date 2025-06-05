import cv2 as cv
import mediapipe as mp

mpHands = mp.solutions.hands # type: ignore
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils # type: ignore
tipIds = [4, 8, 12, 16, 20]

def detect_fingers(img):
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    fingers = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            h, w, _ = img.shape
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            fingers.append(1 if lmList[4][0] > lmList[3][0] else 0)

            for id in range(1, 5):
                fingers.append(1 if lmList[tipIds[id]][1] < lmList[tipIds[id] - 2][1] else 0)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    return sum(fingers), img
