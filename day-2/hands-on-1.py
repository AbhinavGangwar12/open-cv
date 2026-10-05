import cv2
import numpy as np

def assess_and_route(image_path: str) -> str:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError("Image not found")
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    if variance < 100 : return "route_to_sharpening"
    if np.mean(img) < 80: return "route_to_clahe"
    return "route_to_vision_llm"
