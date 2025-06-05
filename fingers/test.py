import cv2 as cv
import mediapipe as mp
import os

# Initialize webcam
cap = cv.VideoCapture(0)

# Fullscreen
cv.namedWindow("Finger Counter", cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty("Finger Counter", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

# MediaPipe Hands
# mp.solutions.hands -> initializes mediapipe's hand detection module
# mpHands.Hands -> Hands is a method of mp.solutions.hands and parameter max_num_hands=1 means at max it will detect 1 hand.
# mp.solutions.drawing_utils -> initializes media pipe's drawing utilities. used for visualization.
mpHands = mp.solutions.hands # type: ignore
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils # type: ignore

# Finger tip IDs (used for counting)
tipIds = [4, 8, 12, 16, 20]

# Colors for each fingertip: thumb to pinky
finger_colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (0, 165, 255)]

# Load and resize finger images (0.jpg to 5.jpg)
folder_path = 'fingers'
finger_images = [
    cv.resize(cv.imread(f"{folder_path}/{i}.jpg"), (100, 100)) 
    if cv.imread(f"{folder_path}/{i}.jpg") is not None else None 
    for i in range(6)
]

while True:
    success, img = cap.read()
    if not success:
        break

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

            # Thumb (assumed right hand)
            fingers.append(1 if lmList[4][0] > lmList[3][0] else 0)

            # 4 Fingers (index to pinky)
            for i in range(1, 5):
                fingers.append(1 if lmList[tipIds[i]][1] < lmList[tipIds[i] - 2][1] else 0)

            # Draw fingertip circles with different colors
            for i, tipId in enumerate(tipIds):
                cx, cy = lmList[tipId]
                cv.circle(img, (cx, cy), 15, finger_colors[i], cv.FILLED)

    totalFingers = sum(fingers)

    # Draw finger count text
    cv.putText(img, f'Detected: {totalFingers}', (50, 100),
               cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    # Overlay finger image (if available)
    if 0 <= totalFingers <= 5 and finger_images[totalFingers] is not None:
        img[0:100, -100:] = finger_images[totalFingers]

    cv.imshow("Finger Counter", img)

    # Exit on ESC key
    if cv.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv.destroyAllWindows()
