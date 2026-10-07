# HANDS-ON 2
import cv2 
import numpy as np

def extract_orb_features(image_path: str, max_features: int = 500) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError("Image not found")

    orb = cv2.ORB_create(nfeatures=max_features)
    _, descriptors = orb.detectAndCompute(img, None)

    if descriptors is None:
        return np.zeros((1, 32), dtype=np.uint8)
    return descriptors


print(extract_orb_features("images/sample.jpg"))

