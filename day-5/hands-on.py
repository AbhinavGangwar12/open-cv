import cv2 
from langchain.tools import tool 

@tool 
def detect_faces_and_report(image_path: str) -> str:
    """Detects faces in an image, finds the largest one, and returns a summary report."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontal_face_default.xml")
    if face_cascade.empty():
        raise RuntimeError("Failed to load cascade classifier XML file.")
        
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    count = len(faces)
    
    if count == 0:
        return "Found 0 faces."
        
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face
    return f"Found {count} faces. The largest face is located at x={x}, y={y} with width={w}, height={h}."

def calculate_iou(box1: list, box2: list) -> float:
    x1, y1, x2, y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    x3, y3, x4, y4 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]

    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    if inter_w == 0 or inter_h == 0:
        return 0.0

    inter_area = inter_w * inter_h
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    if union_area == 0.0:
        return 0.0
    return float(inter_area / union_area)


if __name__ == '__main__':
    # print(detect_faces_and_report.invoke({"image_path": "images/faces.jpg"}))

    print(calculate_iou([0, 0, 2, 2], [5, 5, 2, 2]))