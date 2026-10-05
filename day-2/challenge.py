import cv2 
import numpy as np 

img = cv2.imread("images/sample.jpg", cv2.IMREAD_GRAYSCALE)
if not img: raise FileNotFoundError("Not found")

blur_bil = cv2.bilateralFilter(img, 9, 75, 75)
_, otsu = cv2.threshold(blur_bil, 0, 255 , cv2.THRESH_BINARY | cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
to_save = cv2.erode(otsu, kernel=kernel, iterations=1)
cv2.imwrite("outputs/processed_binary.jpg", to_save)
