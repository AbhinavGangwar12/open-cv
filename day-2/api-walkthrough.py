import cv2 
import matplotlib.pyplot as plt 
import numpy as np

gray = cv2.imread("images/sample.jpg", cv2.IMREAD_GRAYSCALE)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_clahe = clahe.apply(gray)

blurr_gauss = cv2.GaussianBlur(gray, (5,5), 0)
blur_median = cv2.medianBlur(gray, 5)
blur_bilat = cv2.bilateralFilter(gray, 9, 75, 75)

_, thresh_otsu = cv2.threshold(gray,0, 255 ,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


kernel = np.ones((3,3), np.uint8)
eroded = cv2.erode(thresh_otsu, kernel=kernel, iterations=1)
dilate = cv2.dilate(thresh_otsu, kernel=kernel, iterations=1)

edges = cv2.Canny(gray, threshold1=100, threshold2=200)

output_images = [gray_clahe, blurr_gauss, blur_median, blur_bilat, thresh_otsu, thresh_adapt, eroded, dilate, edges]

for i, img in enumerate(output_images):
    cv2.imwrite(f"outputs/day2image{i}.jpg",img)
