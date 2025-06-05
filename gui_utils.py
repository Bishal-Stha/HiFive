import cv2 as cv

def setup_fullscreen(window_name="Finger Counter"):
    cv.namedWindow(window_name, cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

def draw_overlay(img, count, overlay_image):
    cv.putText(img, f'Detected: {count}', (50, 100), cv.FONT_HERSHEY_COMPLEX, 2, (255,0,0), 2)
    if overlay_image is not None:
        img[0:100, -100:] = overlay_image
    return img
