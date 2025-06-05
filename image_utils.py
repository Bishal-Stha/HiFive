import cv2 as cv

def load_finger_images(folder_path='fingers'):
    images = [cv.imread(f"{folder_path}/{i}.png") for i in range(6)]
    return [cv.resize(img, (100, 100)) if img is not None else None for img in images]
