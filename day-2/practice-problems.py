import cv2 
import numpy as np
from langchain.tools import tool 

# 1
@tool 
def clean_document(path: str, output: str):
    """takes a photo of a piece of paper, applies Grayscale -> Median Blur (to remove noise) -> Adaptive Thresholding (to handle uneven lighting), and saves it."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError("Image not found")
    blur_med = cv2.medianBlur(img, 5)
    to_save = cv2.adaptiveThreshold(blur_med, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(output, to_save)
    return

# 2
@tool
def detect_edges(path: str, output:str):
    """ runs a Bilateral Filter (to smooth flat areas but keep edges), then applies Canny Edge detection, and saves the result."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError("Image not found")

    blur_bil = cv2.bilateralFilter(img, 9, 75,75)
    to_save = cv2.Canny(blur_bil, threshold1=100, threshold2=220)
    cv2.imwrite(output, to_save) 
    return

# 3
@tool
def close_gaps(img):
    """takes a binary image, applies a morphological "Closing" operation (cv2.morphologyEx with cv2.MORPH_CLOSE) to connect broken letters in OCR text."""
    _, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU )
    kernel = np.ones((3,3), np.uint8)
    return cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel=kernel)


# ML functions
# 4
def split_and_merge(path: str):
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError("Image not found")

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h , s , v = hsv_img[:,:,0], hsv_img[:,:, 1], hsv_img[:,:, 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)
    hsv_img = cv2.merge([h,s,v])
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

# core
# 6
def masking(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None: 
        raise FileNotFoundError("Image not found")

    h, w = img.shape[:2]
    base = np.zeros((h, w), np.uint8)
    # Fixed center coordinate order (width, height) and set color to 255 (white) filled
    mask = cv2.circle(base, center=(w // 2, h // 2), radius=4, color=255, thickness=-1)
    
    # Correct usage of mask parameter for multi-channel images
    return cv2.bitwise_and(img, img, mask=mask)
