import cv2
import os
import numpy as np
import urllib.request

def blur_faces(img_path: str):
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        print("XML file not found in cv2.data. Downloading a local copy...")
        cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        cascade_path = "haarcascade_frontalface_default.xml"  # Save in current directory
        if not os.path.exists(cascade_path):
            urllib.request.urlretrieve(cascade_url, cascade_path)

    face_cascade = cv2.CascadeClassifier(cascade_path)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    for (x,y,w,h) in faces:
        face = img[y:y+h, x:x+w]
        k_size = (w // 3) | 1
        blurred_face = cv2.GaussianBlur(face, (k_size, k_size), 0)
        img[y:y+h, x:x+w] = blurred_face
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/blurred.jpg", img)

def crop_save_detections(img_path: str, boxes: list[list[float]]):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return
    os.makedirs("outputs/cropped", exist_ok=True)
    paths = []
    for i, box in enumerate(boxes):
        x, y, w, h = map(int, box)
        cropped = img[y:y+h, x:x+w]
        if cropped.size == 0:
            print(f"Warning: Box {i} resulted in an empty crop. Skipping.")
            continue
        path = f"outputs/cropped/crop_{i}.jpg"
        cv2.imwrite(path, cropped)
        paths.append(path)
    return paths

def convert_to_blob(image: np.ndarray, target_size: tuple[int,int] = (416, 416)) -> np.ndarray:
    image = image[:, : , ::-1]
    image = cv2.resize(image, dsize=target_size)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, [2,0,1])
    image = np.expand_dims(image, axis=0)
    return image

def convert_to_standard(cords: tuple[int,int,int,int]) -> tuple[int,int,int,int]:
    center_x, center_y, w, h = cords 
    top_left_x = int(center_x - w / 2)
    top_left_y = int(center_y - h / 2)
    return (top_left_x, top_left_y, w, h)


if __name__ == '__main__':
    blur_faces("images/faces.jpg")
    print("Done.")
