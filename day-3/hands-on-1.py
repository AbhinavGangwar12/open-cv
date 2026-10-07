# HANDS-ON 1
import cv2 


def find_document_corners(image_path : str ) -> list:
    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError("Image not found")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    gaussian = cv2.GaussianBlur(gray, (5,5), 2)
    canny = cv2.Canny(gaussian, threshold1=75, threshold2=200)

    contours , hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c = sorted_contours[0]

    epsilon = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon=epsilon, closed=True)
    if approx.shape[0] == 4:
        return [approx[i, :,:] for i in range(approx.shape[0])]
    return []

print(find_document_corners("images/sample.jpg"))