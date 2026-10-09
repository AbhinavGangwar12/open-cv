import cv2

photo = cv2.imread("images/sample.jpg")
img = cv2.imread("images/sample.jpg", cv2.IMREAD_GRAYSCALE)
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i,c in enumerate(contours):
    if cv2.contourArea(c) > 500:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(photo, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(photo, f"Rect:{i}", (x, max(y - 10, 20)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 200, 0), 3)

cv2.imwrite("outputs/coding-challenge.jpg", photo)
print("Saved")
