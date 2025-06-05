import cv2 as cv
from camera import initialize_camera
from hand_tracker import detect_fingers
from image_utils import load_finger_images
from gui_utils import setup_fullscreen, draw_overlay

def main():
    cap = initialize_camera()
    finger_images = load_finger_images()
    setup_fullscreen()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv.flip(img, 1)
        total_fingers, img = detect_fingers(img)
        img = draw_overlay(img, total_fingers, finger_images[total_fingers] if 0 <= total_fingers <= 5 else None)

        cv.imshow("Finger Counter", img)
        if cv.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
