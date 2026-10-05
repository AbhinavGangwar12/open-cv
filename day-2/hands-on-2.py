import random 
import cv2 
import numpy as np

def augment_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError("Image not found")

    choice = random.random()
    if choice < 0.33:
        return cv2.medianBlur(img, 5)
    
    elif choice > 0.66:
        # creating noise
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        _, img = cv2.threshold(img, 250, 255, cv2.THRESH_TRUNC)
        return np.array(img, dtype=np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    return cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)


