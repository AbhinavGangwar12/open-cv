import cv2 
import numpy as np 

path = "images/sample.jpg"
img = cv2.imread(path)
h, w, c = img.shape 
mid_h , mid_w = h // 2, w //2 

cropped = img[mid_w - 250 : mid_w + 250, mid_h - 250 : mid_h + 250]
cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
try:
    cv2.imwrite("outputs/center_gray.jpg", cropped)
    print("Done.")
except Exception as e:
    pass

h_25 , w_25 = h // 4, w // 4
cropped = img[h_25 : h - h_25, w_25 : w - w_25]