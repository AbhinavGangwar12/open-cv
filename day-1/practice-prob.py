from langchain.tools import tool 
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
import cv2 
import numpy as np

# 1
@tool
def draw_bounding_box(path: str, cords: tuple[int], label: str):
    """ It draws the box and label, then saves the image. (Useful for agents visualizing their detections)."""
    img = cv2.imread(path)
    x, y, w, h = cords
    if img is None: raise FileNotFoundError("Image not found!")
    img_ = img.copy()
    cv2.rectangle(img_, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)
    cv2.putText(img_, label, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 0.7, color=(0, 255, 0), thickness=1)
    cv2.imwrite("outputs/custom.png", img=img_)
    return
# 2
@tool 
def resize_for_token_limit(path: str):
    """takes an image and scales it down so its longest side is exactly 512 pixels, preserving aspect ratio."""
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError("Image not found!")
    h, w, c = img.shape

    target = 512 
    longest = max(h, w)
    scale = target / longest 

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized = np.transpose(resized, (2, 0, 1))
    return resized
# 3
@tool
def check_if_grayscale(path: str):
    """ loads an image and returns True if all 3 channels are identical."""
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError("Image not found!")

    one, two, three = img[:,:,0], img[:,:,1], img[:,:,2]
    return np.array_equal(one, two) and np.array_equal(two, three)
# 4
@tool 
def extract_color_stats(path: str):
    """converts an image to HSV and returns the mean Hue, Saturation, and Value."""
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError("Image not found!")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv, axis=(0,1))

tools = [draw_bounding_box, resize_for_token_limit, check_if_grayscale, extract_color_stats]
llm = ChatMistralAI(model_name="mistral-small-latest") # dummy model
app = create_agent(llm, tools)







