import cv2 
import numpy as np

img = cv2.imread("images/sample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_drawn = img.copy()
cont = cv2.drawContours(img_drawn, contours, -1, (0, 255, 0), 2)

cv2.imwrite("outputs/contours.jpg", img_drawn)

# Finding the area of a contour
c = contours[0]
area = cv2.contourArea(c)
print(area)

# Bounding box 
x, y, h, w = cv2.boundingRect(c)
bounding = cv2.rectangle(img_drawn, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imwrite("outputs/contours.jpg", bounding)

orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)
keypoint = cv2.drawKeypoints(img, keypoints, None, (0,255,0),flags=0)
cv2.imwrite("outputs/keypoints.jpg", keypoint)

