import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read, Convert, Show, Save
img_bgr = cv2.imread('images/sample.jpg') # Returns None if path is wrong. Does NOT throw an error.
if img_bgr is None: raise FileNotFoundError("Image not found!")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
cv2.imwrite('outputs/output1.png', img_bgr) # Always save in BGR, cv2 handles it

# 2. Slicing (ROI) and Drawing
# ROI: img[y1:y2, x1:x2]
roi = img_bgr[50:150, 200:300]
cv2.imwrite('outputs/output2.png', roi) # Always save in BGR, cv2 handles it


# Drawing functions modify the array IN-PLACE.
img_copy = img_bgr.copy()
cv2.rectangle(img_copy, pt1=(200, 50), pt2=(300, 150), color=(0, 255, 0), thickness=2)
cv2.putText(img_copy, "ROI", (200, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.imwrite('outputs/output3.png', img_copy) # Always save in BGR, cv2 handles it

# 3. Geometric Transformations
# Resize
resized = cv2.resize(img_bgr, (224, 224)) # Note: cv2.resize expects (Width, Height)!
cv2.imwrite('outputs/output4.png', resized) # Always save in BGR, cv2 handles it

# Rotate (Affine)
h, w = img_bgr.shape[:2]
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
rotated = cv2.warpAffine(img_bgr, rotation_matrix, (w, h))
cv2.imwrite('outputs/output5.png', rotated) # Always save in BGR, cv2 handles it

# 4. Video / Webcam Reading
cap = cv2.VideoCapture(0) # 0 for default webcam, or 'video.mp4'
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret: break
#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()

