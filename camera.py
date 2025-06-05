import cv2 as cv

def initialize_camera():
    cap = cv.VideoCapture(0)
    return cap
