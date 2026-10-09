# PRACTICE PROBLEMS
from langchain.tools import tool
import cv2 
import math
import numpy as np
#1 
@tool 
def count_objects(image_path: str):
    """takes a binary image path, finds external contours, filters out any contour with an area < 100 (noise), and returns the final count of objects."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError("Image not found!")
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in contours:
        if cv2.contourArea(c) >= 100:
            count+=1
    return count 

#2
@tool
def crop_all_objects(image_path: str):
    """finds all valid contours, calculates their boundingRect, crops each from the original image, and saves them as object_0.jpg, object_1.jpg, etc."""
    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError("Image not found!")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    counter = 0
    for c in contours:
        if cv2.contourArea(c) > 0:
            x, y, w, h = cv2.boundingRect(c)
            cropped = img[y : y + h, x : x + w]
            cv2.imwrite(f"outputs/object_{counter}.jpg", cropped)
            counter += 1

# 3
@tool 
def is_circle(contour):
    """Takes a contour, calculates its area and its perimeter. Compare the area to the theoretical area of a circle with the same perimeter ($Area = \frac{Perimeter^2}{4\pi}$). If they are within 10% of each other, return True."""

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)  # Added required 'closed' parameter
    
    # Correct formula: Perimeter^2 / (4 * pi)
    theoretical_area = (perimeter ** 2) / (4 * math.pi)
    
    lower = theoretical_area - (theoretical_area * 0.1)
    upper = theoretical_area + (theoretical_area * 0.1)
    
    if lower <= area <= upper:
        return True
    return False

# 4
@tool
def find_icon(screen_path: str, icon_path: str):
    """ returns the top-left (x,y) coordinates of the icon on the screen."""
    screen = cv2.imread(screen_path, cv2.IMREAD_GRAYSCALE)
    if screen is None: raise FileNotFoundError("Image not found")
    icon = cv2.imread(icon_path, cv2.IMREAD_GRAYSCALE)
    if icon is None: raise FileNotFoundError("Image not found")

    res = cv2.matchTemplate(screen, icon, cv2.TM_CCOEFF_NORMED)
    _, _, _, top_left = cv2.minMaxLoc(res)
    return top_left


# 5
def centroid(c):
    M = cv2.moments(c)
    if M["00"] != 0:
        cX = M["10"] / M["00"]
        cY = M["01"] / M["00"]
        return cX, cY 

    return (0,0)

# 6

# FATAL FLAW FIX: Must be a single-channel image for findContours, not 3 channels.
img = np.zeros((500, 500), dtype=np.uint8)

# Draw a filled white triangle
points = np.array([[250, 100], [100, 400], [400, 400]], dtype=np.int32)
cv2.fillPoly(img, [points], color=255)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    c = contours[0]
    
    # FATAL FLAW FIX: Pass 'c' (the array), not 'contours' (the list).
    perimeter = cv2.arcLength(c, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(c, epsilon, True)
    
    print(f"Points found: {len(approx)}")
    if len(approx) == 3:
        print("YES")
    else:
        print("NO")
    
